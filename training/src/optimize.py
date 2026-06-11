import optuna
import json
import torch
import torch.nn as nn
import os
import numpy as np
import pickle
from sklearn.model_selection import KFold
from core.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, DecoderLoss, PhysicsInformedLoss, TorchScaler, convert_v_to_f_torch
from .data_loader import format_data
from core.physics import get_geometry
from tqdm import tqdm

def optimize(df_train, entree, residuelle, inter, suffixe, n_trials=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    
    os.makedirs("hyperparametres", exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f" OPTIMISATION MODÈLE PRÉDICTIF : {saved_name}")
    print(f"{'='*50}")
    
    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    is_cnn = (entree == 'GM')

    # Récupération des moyennes SVEN globales pour le calcul du vrai (%) ---
    global_mean_fn = df_train['Fn_SVEN'].abs().mean()
    global_mean_ft = df_train['Ft_SVEN'].abs().mean()

    # INITIALISATION SÉCURISÉE DE LA VARIABLE POUR ÉVITER LE NAMEERROR
    V_BEM_phys_full = None

    # --- PRÉPARATION GÉOMÉTRIE ET SCALERS ---
    if inter == 'v':
        # Force la création du scaler_f
        _, _ = format_data(df_train, entree, residuelle, 'f', is_train=True, device=device)
        
        with open(f"training/scalers/scaler_Y_{entree}_{residuelle}_v.pkl", 'rb') as f: scaler_v = pickle.load(f)
        with open(f"training/scalers/scaler_Y_{entree}_{residuelle}_f.pkl", 'rb') as f: scaler_f = pickle.load(f)
        scaler_v_torch = TorchScaler(scaler_v, device)
        scaler_f_torch = TorchScaler(scaler_f, device)
        
        polar_surrogate = PolarSurrogate(device=device).to(device)

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

        # Création du tenseur V_BEM_phys pour la stratégie 1_v
        if str(residuelle) == '1':
            _, Y_full_abs = format_data(df_train, entree, '0', inter, is_train=True, device=device)
            
            # Utilisation du scaler ABSOLU pour dénormaliser les cibles absolues
            with open(f"training/scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs: 
                scaler_v_abs = pickle.load(f_abs)
            scaler_v_abs_torch = TorchScaler(scaler_v_abs, device)
            
            V_SVEN_phys_full = scaler_v_abs_torch.inverse_transform(Y_full_abs)
            Delta_V_phys_full = scaler_v_torch.inverse_transform(Y_full)
            V_BEM_phys_full = V_SVEN_phys_full - Delta_V_phys_full
    else:
        with open(f"training/scalers/scaler_Y_{entree}_{residuelle}_f.pkl", 'rb') as f: scaler_f = pickle.load(f)
        scaler_f_torch = TorchScaler(scaler_f, device)

    
    # --- 1. Gestion de l'Auto-encodeur ---
    Y_train_target = Y_full  
    
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

    # --- 2. Objectif Optuna ---
    def objective_model(trial):
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.4)
        n_layers = trial.suggest_int('n_layers', 2, 5)
        
        if entree == 'GV':
            n_neurons = trial.suggest_int('n_neurons', 128, 512, step=64)
        elif entree == 'GM':
            base_filters = trial.suggest_categorical('base_filters', [16, 32, 64])

        if inter == 'v':
            lambda_ingenieur = 0.5
            criterion = PhysicsInformedLoss(current_ae, scaler_v, scaler_f, lambda_ingenieur, r_tensor, c_tensor, polar_surrogate, device)
        else:
            criterion = DecoderLoss(current_ae)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        cv_phys_scores = [] 
        
        for train_idx, val_idx in kf.split(X_full.cpu().numpy()):
            X_tr, Y_tr = X_full[train_idx], Y_train_target[train_idx]
            X_val, Y_val = X_full[val_idx], Y_train_target[val_idx]
            
            # Utilisation sécurisée de la variable (vaut None ou le tenseur)
            v_bem_tr = V_BEM_phys_full[train_idx] if V_BEM_phys_full is not None else None
            v_bem_val = V_BEM_phys_full[val_idx] if V_BEM_phys_full is not None else None
            
            if entree == 'GV':
                target_dim = latent_dim if use_ae else Y_full.shape[1]
                model = TurbineMLP(X_full.shape[1], target_dim, n_layers, n_neurons, dropout_rate, device=device).to(device)
            elif entree == 'GM':
                target_dim = latent_dim if use_ae else Y_full.shape[1]
                model = TurbineCNN(in_channels=X_full.shape[1], out_channels=target_dim, 
                                   use_autoencoder=use_ae, latent_dim=latent_dim,
                                   n_layers=n_layers, base_filters=base_filters, dropout_rate=dropout_rate, device=device).to(device)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_val_loss = float('inf')

            for epoch in range(150):
                model.train()
                optimizer.zero_grad()

                if inter == 'v' and V_BEM_phys_full is not None:
                    loss = criterion(model(X_tr), Y_tr, v_bem_phys=v_bem_tr)
                else:
                    loss = criterion(model(X_tr), Y_tr)

                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    preds_val = model(X_val)
                    val_loss = criterion(preds_val, Y_val).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
            cv_scores.append(best_val_loss)
            
            # --- CALCUL DU TOTAL_SCORE (%) PHYSIQUE SUR LA VALIDATION ---
            model.eval()
            with torch.no_grad():
                preds_norm = current_ae.decode(model(X_val)) if use_ae else model(X_val)
                
                if inter == 'v':
                    v_pred_phys = scaler_v_torch.inverse_transform(preds_norm)
                    v_true_phys = scaler_v_torch.inverse_transform(Y_val)
                    
                    if v_bem_val is not None:
                        v_pred_phys = v_pred_phys + v_bem_val
                        v_true_phys = v_true_phys + v_bem_val

                    if is_cnn:
                        v_eff_p, alpha_p = v_pred_phys[:, 0], v_pred_phys[:, 1]
                        v_eff_t, alpha_t = v_true_phys[:, 0], v_true_phys[:, 1]
                    else:
                        v_eff_p, alpha_p = v_pred_phys[:, 0::2], v_pred_phys[:, 1::2]
                        v_eff_t, alpha_t = v_true_phys[:, 0::2], v_true_phys[:, 1::2]
                        
                    f_pred_phys = convert_v_to_f_torch(v_eff_p, alpha_p, r_tensor, c_tensor, polar_surrogate)
                    f_true_phys = convert_v_to_f_torch(v_eff_t, alpha_t, r_tensor, c_tensor, polar_surrogate)
                    
                    if is_cnn:
                        # Alignement des axes comme dans la loss customisée
                        f_pred_phys = f_pred_phys.permute(0, 3, 1, 2)
                        f_true_phys = f_true_phys.permute(0, 3, 1, 2)
                        Fn_p, Ft_p = f_pred_phys[:, 0], f_pred_phys[:, 1]
                        Fn_t, Ft_t = f_true_phys[:, 0], f_true_phys[:, 1]
                    else:
                        Fn_p, Ft_p = f_pred_phys[..., 0], f_pred_phys[..., 1]
                        Fn_t, Ft_t = f_true_phys[..., 0], f_true_phys[..., 1]
                else:
                    f_pred_phys = scaler_f_torch.inverse_transform(preds_norm)
                    f_true_phys = scaler_f_torch.inverse_transform(Y_val)
                    
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
                
                cv_phys_scores.append((rel_fn + rel_ft).item())

        trial.set_user_attr("cv_score_phys_percent", sum(cv_phys_scores) / len(cv_phys_scores))
        return sum(cv_scores) / len(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_model = optuna.create_study(direction='minimize')
    study_model.optimize(objective_model, n_trials=n_trials, show_progress_bar=True)
    
    # --- 3. Sauvegarde ---
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