import optuna
import json
import torch
import os
import numpy as np
import pickle
from core.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, DecoderLoss, PhysicsInformedLoss, TorchScaler, convert_v_to_f_torch
from training.src.data_loader import format_data, get_D_tensor, get_V_app_tensor
from training.src.trainer import cross_validate
from core.physics import get_geometry
from core.config import EPOCHS_OPTUNA, CV_SPLITS, LR_BOUNDS, DROPOUT_BOUNDS, MLP_LAYERS_BOUNDS, MLP_NEURONS_CHOICES, CNN_LAYERS_BOUNDS, CNN_FILTERS_CHOICES, LAMBDA_INGENIEUR

def optimize(df_train, entree, residuelle, inter, suffixe, n_trials=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    
    os.makedirs("training/hyperparametres", exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f" OPTIMISATION MODÈLE PRÉDICTIF : {saved_name}")
    print(f"{'='*50}")
    
    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    is_cnn = (entree == 'GM')

    # Récupération des moyennes SVEN globales pour le calcul du vrai (%) relatif
    global_mean_fn = df_train['Fn_SVEN'].abs().mean()
    global_mean_ft = df_train['Ft_SVEN'].abs().mean()

    # Charger le scaler global (qui sert pour f ET pour v)
    with open(f"training/scalers/scaler_Y_{base_model_name}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
    scaler_Y_torch = TorchScaler(scaler_Y, device)

    V_BEM_phys_full = None
    V_app_full = None
    D_full = None

    # --- PRÉPARATION DE L'ÉVALUATION PHYSIQUE ---
    if inter == 'v':
        _, _ = format_data(df_train, entree, residuelle, 'f', is_train=True, device=device)
        with open(f"training/scalers/scaler_Y_{entree}_{residuelle}_f.pkl", 'rb') as f:
            scaler_f = pickle.load(f)
            
        polar_surrogate = PolarSurrogate(device=device).to(device)
        V_app_full = get_V_app_tensor(df_train, entree, device)

        geom = get_geometry()
        if is_cnn:
            r_uniques = np.sort(df_train['r'].unique())
            theta_uniques = np.sort(df_train['theta'].unique())
            R_grid, _ = np.meshgrid(r_uniques, theta_uniques, indexing='ij')
            r_tensor = torch.tensor(R_grid, dtype=torch.float32, device=device)
            c_grid = np.array([geom.get_chord(r) for r in r_uniques])
            C_grid, _ = np.meshgrid(c_grid, theta_uniques, indexing='ij')
            c_tensor = torch.tensor(C_grid, dtype=torch.float32, device=device)
        else:
            group = df_train[(df_train['yaw'] == df_train['yaw'].iloc[0])]
            if 'TSR' in group.columns: group = group[group['TSR'] == group['TSR'].iloc[0]]
            group = group.sort_values(['theta', 'r'])
            r_array = group['r'].values
            r_tensor = torch.tensor(r_array, dtype=torch.float32, device=device)
            c_tensor = torch.tensor(np.array([geom.get_chord(r) for r in r_array]), dtype=torch.float32, device=device)

        # Création du tenseur V_BEM_phys_full (qui contient an_bem et at_bem) pour la strat 1_v ou 2_v
        if str(residuelle) in ['1', '2']:
            _, Y_full_abs_scaled = format_data(df_train, entree, '0', inter, is_train=False, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs: 
                scaler_v_abs = pickle.load(f_abs)
                
            an_at_sven = TorchScaler(scaler_v_abs, device).inverse_transform(Y_full_abs_scaled)
            an_at_delta = scaler_Y_torch.inverse_transform(Y_full)
            V_BEM_phys_full = an_at_sven - an_at_delta
    else:
        D_full = get_D_tensor(df_train, entree, device)

    
    # --- CHARGEMENT DE L'AUTO-ENCODEUR (SI NÉCESSAIRE) ---
    if suffixe == 'D0':
        use_ae = False
        latent_dim = 0
        ae_params = {"use_autoencoder": False, "latent_dim": 0}
        current_ae = None
    else:
        ae_master_path = "training/hyperparametres/ae_hyperparameters.json"
        ae_weights_path = f"training/models/ae/ae_{saved_name}.pth"
        
        use_ae = os.path.exists(ae_master_path) and os.path.exists(ae_weights_path)
        if use_ae:
            with open(ae_master_path, "r") as f: all_ae_params = json.load(f)
            if saved_name in all_ae_params:
                ae_params = all_ae_params[saved_name]
                latent_dim = ae_params['latent_dim']
                
                if entree == 'GM':
                    current_ae = ConvolutionalAutoencoder(in_channels=Y_full.shape[1], latent_dim=latent_dim,
                                                        depth=ae_params['ae_depth'], base_filters=ae_params['ae_base_filters'], device=device).to(device)
                else:
                    current_ae = LinearAutoencoder(in_features=Y_full.shape[1], latent_dim=latent_dim, device=device).to(device)
                    
                current_ae.load_state_dict(torch.load(ae_weights_path, map_location=device))
                current_ae.eval()
            else:
                use_ae = False
                
        if not use_ae:
            latent_dim = 0
            ae_params = {"use_autoencoder": False, "latent_dim": 0}
            current_ae = None

    # --- DÉFINITION DE LA MÉTRIQUE PHYSIQUE POUR OPTUNA ---
    def compute_phys_score(model, X_val, Y_val, val_idx, preds_val):
        """ Évalue en N/m la performance pour la Cross-Validation Optuna """
        preds_norm = current_ae.decode(preds_val) if use_ae else preds_val
        
        if inter == 'v':
            coeffs_pred = scaler_Y_torch.inverse_transform(preds_norm)
            coeffs_true = scaler_Y_torch.inverse_transform(Y_val)
            v_bem_val = V_BEM_phys_full[val_idx] if V_BEM_phys_full is not None else None

            if v_bem_val is not None:
                coeffs_pred = coeffs_pred + v_bem_val
                coeffs_true = coeffs_true + v_bem_val

            if is_cnn:
                an_p, at_p = coeffs_pred[:, 0], coeffs_pred[:, 1]
                an_t, at_t = coeffs_true[:, 0], coeffs_true[:, 1]
            else:
                an_p, at_p = coeffs_pred[:, 0::2], coeffs_pred[:, 1::2]
                an_t, at_t = coeffs_true[:, 0::2], coeffs_true[:, 1::2]

            # Cartésien vers Polaire (alpha en degrés)
            alpha_p_deg = torch.atan2(an_p, at_p) * (180.0 / torch.pi)
            alpha_t_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)

            # V_eff = V_app * sqrt(an^2 + at^2)
            v_app_slice = V_app_full[val_idx]
            v_eff_p = v_app_slice * torch.sqrt(an_p**2 + at_p**2)
            v_eff_t = v_app_slice * torch.sqrt(an_t**2 + at_t**2)

            f_pred_phys = convert_v_to_f_torch(v_eff_p, alpha_p_deg, r_tensor, c_tensor, polar_surrogate)
            f_true_phys = convert_v_to_f_torch(v_eff_t, alpha_t_deg, r_tensor, c_tensor, polar_surrogate)

            if is_cnn:
                Fn_p, Ft_p = f_pred_phys[..., 0], f_pred_phys[..., 1]
                Fn_t, Ft_t = f_true_phys[..., 0], f_true_phys[..., 1]
            else:
                Fn_p, Ft_p = f_pred_phys[..., 0], f_pred_phys[..., 1]
                Fn_t, Ft_t = f_true_phys[..., 0], f_true_phys[..., 1]

        else: # inter == 'f'
            coeffs_pred = scaler_Y_torch.inverse_transform(preds_norm)
            coeffs_true = scaler_Y_torch.inverse_transform(Y_val)

            D_val = D_full[val_idx]
            f_pred_phys = coeffs_pred * D_val
            f_true_phys = coeffs_true * D_val

            if is_cnn:
                Fn_p, Ft_p = f_pred_phys[:, 0], f_pred_phys[:, 1]
                Fn_t, Ft_t = f_true_phys[:, 0], f_true_phys[:, 1]
            else:
                Fn_p, Ft_p = f_pred_phys[:, 0::2], f_pred_phys[:, 1::2]
                Fn_t, Ft_t = f_true_phys[:, 0::2], f_true_phys[:, 1::2]

        rmse_fn = torch.sqrt(torch.mean((Fn_p - Fn_t)**2))
        rmse_ft = torch.sqrt(torch.mean((Ft_p - Ft_t)**2))

        rel_fn = (rmse_fn / global_mean_fn * 100) if global_mean_fn > 0 else 0
        rel_ft = (rmse_ft / global_mean_ft * 100) if global_mean_ft > 0 else 0

        return (rel_fn + rel_ft).item()

    # --- OBJECTIF OPTUNA ---
    def objective_model(trial):
        lr = trial.suggest_float('lr', LR_BOUNDS[0], LR_BOUNDS[1], log=True)
        dropout_rate = trial.suggest_float('dropout_rate', DROPOUT_BOUNDS[0], DROPOUT_BOUNDS[1])
        
        if entree == 'GV':
            model_class = TurbineMLP
            model_kwargs = {
                'input_dim': X_full.shape[1],
                'output_dim': latent_dim if use_ae else Y_full.shape[1],
                'n_layers': trial.suggest_int('n_layers', MLP_LAYERS_BOUNDS[0], MLP_LAYERS_BOUNDS[1]),
                'n_neurons': trial.suggest_categorical('n_neurons', MLP_NEURONS_CHOICES),
                'dropout_rate': dropout_rate,
                'device': device
            }
        elif entree == 'GM':
            model_class = TurbineCNN
            model_kwargs = {
                'in_channels': X_full.shape[1],
                'out_channels': latent_dim if use_ae else Y_full.shape[1],
                'use_autoencoder': use_ae,
                'latent_dim': latent_dim,
                'n_layers': trial.suggest_int('n_layers', CNN_LAYERS_BOUNDS[0], CNN_LAYERS_BOUNDS[1]),
                'base_filters': trial.suggest_categorical('base_filters', CNN_FILTERS_CHOICES),
                'dropout_rate': dropout_rate,
                'device': device
            }

        def criterion_builder(train_idx=None, val_idx=None):
            if inter == 'v':
                v_train = V_app_full[train_idx] if train_idx is not None else V_app_full
                v_val = V_app_full[val_idx] if val_idx is not None else V_app_full
                return PhysicsInformedLoss(current_ae, scaler_Y, scaler_f, LAMBDA_INGENIEUR, r_tensor, c_tensor, polar_surrogate, device, v_app_train=v_train, v_app_val=v_val)
            else:
                return DecoderLoss(current_ae)

        # Délégation de l'entraînement au trainer.py
        mean_val_loss, mean_custom_score, _ = cross_validate(
            X_full=X_full, Y_full=Y_full,
            model_class=model_class, model_kwargs=model_kwargs,
            criterion_builder=criterion_builder,
            epochs=EPOCHS_OPTUNA, lr=lr,
            n_splits=CV_SPLITS, device=device, inter=inter,
            v_bem_phys_full=V_BEM_phys_full,
            compute_metrics_fn=compute_phys_score
        )

        trial.set_user_attr("cv_score_phys_percent", mean_custom_score)
        return mean_val_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_model = optuna.create_study(direction='minimize')
    study_model.optimize(objective_model, n_trials=n_trials, show_progress_bar=True)
    
    # --- SAUVEGARDE DES RÉSULTATS ---
    best_cv_phys = study_model.best_trial.user_attrs["cv_score_phys_percent"]
    final_params = {**ae_params, **study_model.best_params}
    final_params["Total_Score_CV"] = best_cv_phys 
    
    target_json = f"training/hyperparametres/{entree.lower()}_hyperparameters.json"
    if os.path.exists(target_json):
        with open(target_json, "r") as f: all_model_params = json.load(f)
    else:
        all_model_params = {}
        
    all_model_params[saved_name] = final_params
    with open(target_json, "w") as f: json.dump(all_model_params, f, indent=4)
        
    print(f"   [OK] Modèle {saved_name} optimisé.")
    print(f"   -> MSE Validation : {study_model.best_value:.6f}")
    print(f"   -> Erreur Relative Physique (CV) : {best_cv_phys:.2f} %")