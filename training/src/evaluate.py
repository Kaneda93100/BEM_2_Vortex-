import os
import json
import pickle
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from core.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, DecoderLoss, PhysicsInformedLoss, TorchScaler, convert_v_to_f_torch
from training.src.data_loader import format_data, get_D_tensor, get_V_app_tensor
from training.src.trainer import fit_model, cross_validate
from core.physics import convert_v_to_f, get_geometry, compute_dynamic_pressure_D
from core.config import EPOCHS_FINAL, CV_SPLITS, LAMBDA_INGENIEUR

def reconstruct_predictions(df_test, preds, entree, residuelle, inter):
    """ Réaligne les prédictions (déjà décodées et en dimensions physiques) selon la topologie d'origine. """
    res_str = str(residuelle)
    if inter == 'f':
        c1, c2 = 'Fn', 'Ft'
        c1_bem, c2_bem = 'Fn_BEM', 'Ft_BEM'
    elif inter == 'v':
        c1, c2 = 'an', 'at'
        c1_bem, c2_bem = 'an_BEM', 'at_BEM'
        
    records = []
    for i, (y_val, group) in enumerate(df_test.groupby('yaw')):
        if entree == 'GV':
            group = group.sort_values(['theta', 'r'])
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2]
        elif entree == 'GM':
            group = group.sort_values(['r', 'theta'])
            num_r = len(group['r'].unique())
            num_theta = len(group['theta'].unique())
            pred_img = preds[i].reshape(2, num_r, num_theta)
            p_v1, p_v2 = pred_img[0].flatten(), pred_img[1].flatten()
            
        for j, (_, row) in enumerate(group.iterrows()):
            v1, v2 = p_v1[j], p_v2[j]
            if res_str == '1':
                v1 += row[c1_bem]
                v2 += row[c2_bem]
                
            records.append({
                'r': row['r'], 'theta': row['theta'], 'yaw': row['yaw'], 
                f'{c1}_pred': v1, f'{c2}_pred': v2, 
                'Fn_SVEN': row['Fn_SVEN'], 'Ft_SVEN': row['Ft_SVEN']
            })
                
    df_preds = pd.DataFrame(records)
    return pd.merge(df_test, df_preds, on=['r', 'theta', 'yaw', 'Fn_SVEN', 'Ft_SVEN'])

def evaluator(df_train, df_test, entree, residuelle, inter, suffixe):
    V_BEM_phys_train = None
    V_app_full_train = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    recap_path = "training/performance/recap_scores_globaux.csv"
    is_cnn = (entree == 'GM')
    
    print(f"\n{'='*50}")
    print(f" ÉVALUATION EXHAUSTIVE ({CV_SPLITS} Folds CV | Fixe {EPOCHS_FINAL} Epochs) : {saved_name}")
    print(f"{'='*50}")
    
    hp_path = f"training/hyperparametres/{entree.lower()}_hyperparameters.json"
    if not os.path.exists(hp_path): return
        
    with open(hp_path, "r") as f: all_hps = json.load(f)
    if saved_name not in all_hps: return
    best_params = all_hps[saved_name]
        
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=False, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)

    # Charger le scaler global 
    with open(f"training/scalers/scaler_Y_{base_model_name}.pkl", 'rb') as f: 
        scaler_Y = pickle.load(f)
    scaler_Y_torch = TorchScaler(scaler_Y, device)

    # =========================================================================
    # INITIALISATION DES RÉFÉRENCES PHYSIQUES POUR LA CV
    # =========================================================================
    global_mean_fn = df_train['Fn_SVEN'].abs().mean()
    global_mean_ft = df_train['Ft_SVEN'].abs().mean()

    if inter == 'v':
        _, _ = format_data(df_train, entree, residuelle, 'f', is_train=True, device=device)
        with open(f"training/scalers/scaler_Y_{entree}_{residuelle}_f.pkl", 'rb') as f: scaler_f = pickle.load(f)
        
        polar_surrogate = PolarSurrogate(device=device).to(device)
        V_app_full_train = get_V_app_tensor(df_train, entree, device)
        D_train_full = None

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

        if str(residuelle) in ['1', '2']:
            _, Y_full_abs_scaled = format_data(df_train, entree, '0', inter, is_train=False, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs: scaler_v_abs = pickle.load(f_abs)
            
            an_at_sven = TorchScaler(scaler_v_abs, device).inverse_transform(Y_full_abs_scaled)
            an_at_delta = scaler_Y_torch.inverse_transform(Y_train)
            V_BEM_phys_train = an_at_sven - an_at_delta
    else:
        D_train_full = get_D_tensor(df_train, entree, device)
        D_test_full = get_D_tensor(df_test, entree, device)

    # --- CHARGEMENT DE L'AUTO-ENCODEUR ---
    use_ae = best_params.get('use_autoencoder', False) and suffixe != 'D0'
    if use_ae:
        latent_dim = best_params['latent_dim']
        ae_configs = json.load(open("training/hyperparametres/ae_hyperparameters.json", "r"))
        ae_config = ae_configs[saved_name]
        if entree == 'GM':
            current_ae = ConvolutionalAutoencoder(in_channels=Y_train.shape[1], latent_dim=latent_dim, 
                                                  depth=ae_config['ae_depth'], base_filters=ae_config['ae_base_filters'], device=device).to(device)
        else:
            current_ae = LinearAutoencoder(in_features=Y_train.shape[1], latent_dim=latent_dim, device=device).to(device)
        current_ae.load_state_dict(torch.load(f"training/models/ae/ae_{saved_name}.pth", map_location=device))
        current_ae.eval()
    else:
        latent_dim = 0
        current_ae = None

    target_dim = latent_dim if use_ae else Y_train.shape[1]

    def criterion_builder(train_idx=None, val_idx=None):
        if inter == 'v':
            v_train = V_app_full_train[train_idx] if train_idx is not None else V_app_full_train
            v_val = V_app_full_train[val_idx] if val_idx is not None else V_app_full_train
            return PhysicsInformedLoss(current_ae, scaler_Y, scaler_f, LAMBDA_INGENIEUR, r_tensor, c_tensor, polar_surrogate, device, v_app_train=v_train, v_app_val=v_val)
        else:
            return DecoderLoss(current_ae)

    def compute_phys_score(model, X_val, Y_val, val_idx, preds_val):
        """ Calculateur de score physique injecté pour la Cross-Validation """
        preds_norm = current_ae.decode(preds_val) if use_ae else preds_val
        coeffs_pred = scaler_Y_torch.inverse_transform(preds_norm)
        coeffs_true = scaler_Y_torch.inverse_transform(Y_val)
        
        if inter == 'v':
            v_bem_val = V_BEM_phys_train[val_idx] if V_BEM_phys_train is not None else None
            if v_bem_val is not None:
                coeffs_pred += v_bem_val
                coeffs_true += v_bem_val
            an_p, at_p = (coeffs_pred[:, 0], coeffs_pred[:, 1]) if is_cnn else (coeffs_pred[:, 0::2], coeffs_pred[:, 1::2])
            an_t, at_t = (coeffs_true[:, 0], coeffs_true[:, 1]) if is_cnn else (coeffs_true[:, 0::2], coeffs_true[:, 1::2])
            
            alpha_p_deg = torch.atan2(an_p, at_p) * (180.0 / torch.pi)
            alpha_t_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)
            v_eff_p = V_app_full_train[val_idx] * torch.sqrt(an_p**2 + at_p**2)
            v_eff_t = V_app_full_train[val_idx] * torch.sqrt(an_t**2 + at_t**2)
            
            f_pred_phys = convert_v_to_f_torch(v_eff_p, alpha_p_deg, r_tensor, c_tensor, polar_surrogate)
            f_true_phys = convert_v_to_f_torch(v_eff_t, alpha_t_deg, r_tensor, c_tensor, polar_surrogate)
            Fn_p, Ft_p = f_pred_phys[..., 0], f_pred_phys[..., 1]
            Fn_t, Ft_t = f_true_phys[..., 0], f_true_phys[..., 1]
        else:
            D_val = D_train_full[val_idx]
            f_pred_phys = coeffs_pred * D_val
            f_true_phys = coeffs_true * D_val
            Fn_p, Ft_p = (f_pred_phys[:, 0], f_pred_phys[:, 1]) if is_cnn else (f_pred_phys[:, 0::2], f_pred_phys[:, 1::2])
            Fn_t, Ft_t = (f_true_phys[:, 0], f_true_phys[:, 1]) if is_cnn else (f_true_phys[:, 0::2], f_true_phys[:, 1::2])

        rmse_fn = torch.sqrt(torch.mean((Fn_p - Fn_t)**2))
        rmse_ft = torch.sqrt(torch.mean((Ft_p - Ft_t)**2))
        return ((rmse_fn / global_mean_fn * 100) + (rmse_ft / global_mean_ft * 100)).item()

    # =========================================================================
    # PHASE 1 : VALIDATION CROISÉE SUR CV_SPLITS FOLDS
    # =========================================================================
    print(f"\n   [1/2] Lancement de la Cross-Validation ({CV_SPLITS} Folds)...")
    if entree == 'GV':
        model_class = TurbineMLP
        model_kwargs = {'input_dim': X_train.shape[1], 'output_dim': target_dim, 'n_layers': best_params['n_layers'], 'n_neurons': best_params['n_neurons'], 'dropout_rate': best_params['dropout_rate'], 'device': device}
    else:
        model_class = TurbineCNN
        model_kwargs = {'in_channels': X_train.shape[1], 'out_channels': target_dim, 'use_autoencoder': use_ae, 'latent_dim': latent_dim, 'n_layers': best_params['n_layers'], 'base_filters': best_params['base_filters'], 'dropout_rate': best_params['dropout_rate'], 'device': device}

    _, mean_cv_1000, std_cv_1000 = cross_validate(
        X_full=X_train, Y_full=Y_train, model_class=model_class, model_kwargs=model_kwargs,
        criterion_builder=criterion_builder, epochs=EPOCHS_FINAL, lr=best_params['lr'],
        n_splits=CV_SPLITS, device=device, inter=inter, v_bem_phys_full=V_BEM_phys_train,
        compute_metrics_fn=compute_phys_score
    )

    # =========================================================================
    # PHASE 2 : ENTRAÎNEMENT DU MODÈLE FINAL
    # =========================================================================
    print(f"\n   [2/2] Entraînement complet du Modèle Final sur {EPOCHS_FINAL} époques...")
    model_final = model_class(**model_kwargs).to(device)
    model_final, _ = fit_model(
        model=model_final, X=X_train, Y=Y_train, criterion=criterion_builder(None,None), 
        epochs=EPOCHS_FINAL, lr=best_params['lr'], device=device, inter=inter,
        v_bem_phys=V_BEM_phys_train, show_progress=True
    )

    # =========================================================================
    # PHASE 3 : TEST & CALCULS FINAUX
    # =========================================================================
    model_final.eval()
    with torch.no_grad(): 
        preds_raw = model_final(X_test)
        preds_norm_out = current_ae.decode(preds_raw) if use_ae else preds_raw

    preds_norm_np = preds_norm_out.cpu().numpy()
    preds_flat = preds_norm_np.reshape(preds_norm_np.shape[0], -1) if is_cnn else preds_norm_np
    preds_coeffs = scaler_Y.inverse_transform(preds_flat)

    if inter == 'v':
        from core.physics import compute_V_app
        df_test_calc = df_test.copy()
        
        V_app_test_init = compute_V_app(df_test_calc)
        alpha_bem_rad = np.radians(df_test_calc['alpha_BEM'].values)
        
        df_test_calc['an_BEM'] = np.sin(alpha_bem_rad) * df_test_calc['V_eff_BEM'].values / V_app_test_init
        df_test_calc['at_BEM'] = np.cos(alpha_bem_rad) * df_test_calc['V_eff_BEM'].values / V_app_test_init

        df_res = reconstruct_predictions(df_test_calc, preds_coeffs, entree, residuelle, inter)
        
        V_app_test = compute_V_app(df_res)
        
        an_p, at_p = df_res['an_pred'].values, df_res['at_pred'].values
        
        # Inversion trigonométrique différentiable vers l'espace polaire
        alpha_p_deg = np.arctan2(an_p, at_p) * (180.0 / np.pi)
        v_eff_p = V_app_test * np.sqrt(an_p**2 + at_p**2)
        
        df_res['V_eff_pred'] = v_eff_p
        df_res['alpha_pred'] = alpha_p_deg
        
        # Calcul final des forces physiques en N/m
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    
    else:
        D_test_np = D_test_full.cpu().numpy()
        D_test_flat = D_test_np.reshape(D_test_np.shape[0], -1) if is_cnn else D_test_np
        preds_denorm = preds_coeffs * D_test_flat
        df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)

    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    
    # 1. Métriques Physiques (N/m)
    rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    score_total_test = rel_fn + rel_ft
    wd_score = wasserstein_distance(Fn_s, Fn_p) + wasserstein_distance(Ft_s, Ft_p)
    
    # 2. Métriques Normalisées sans unité (Fn/D et Ft/D)
    D_res = compute_dynamic_pressure_D(df_res)
    Fn_s_norm, Ft_s_norm = Fn_s / D_res, Ft_s / D_res
    Fn_p_norm, Ft_p_norm = Fn_p / D_res, Ft_p / D_res
    
    rmse_fn_norm = np.sqrt(np.mean((Fn_p_norm - Fn_s_norm)**2))
    rmse_ft_norm = np.sqrt(np.mean((Ft_p_norm - Ft_s_norm)**2))
    rel_fn_norm = (rmse_fn_norm / np.mean(np.abs(Fn_s_norm))) * 100 if np.mean(np.abs(Fn_s_norm)) != 0 else 0
    rel_ft_norm = (rmse_ft_norm / np.mean(np.abs(Ft_s_norm))) * 100 if np.mean(np.abs(Ft_s_norm)) != 0 else 0
    score_total_bis = rel_fn_norm + rel_ft_norm

    # Dictionnaire Recap global CSV
    results_detail = {
        "Modele": saved_name, "Strat_Entree": entree, "Residuelle": residuelle, "Intermediaire": inter, "Suffixe": suffixe,
        "RMSE_Fn": round(rmse_fn, 4), "RMSE_Ft": round(rmse_ft, 4),
        "RMSE_Fn_norm": round(rmse_fn_norm, 4), "RMSE_Ft_norm": round(rmse_ft_norm, 4),
        "Rel_Fn (%)": round(rel_fn, 2), "Rel_Ft (%)": round(rel_ft, 2),
        "Rel_Fn_norm (%)": round(rel_fn_norm, 2), "Rel_Ft_norm (%)": round(rel_ft_norm, 2),
        "Total_Score_Test (%)": round(score_total_test, 2),
        "Total_Score_Test_Bis (%)": round(score_total_bis, 2),
        "Total_Score_CV_150 (%)": round(best_params.get("Total_Score_CV", -1.0), 2),
        "Total_Score_CV2_1000 (%)": round(mean_cv_1000, 2), 
        "CV2_Variance (%)": round(std_cv_1000, 2),
        "Wasserstein_Dist": round(wd_score, 4)
    }
    
    os.makedirs("training/performance", exist_ok=True)
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != saved_name]
        cols_to_drop = ["Total_Score_CV2_Best_Epoch", "Total_Score_CV2_Best_1.5Epoch", "Best_Epoch"]
        df_recap = df_recap.drop(columns=[c for c in cols_to_drop if c in df_recap.columns])
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
        
    df_recap.to_csv(recap_path, index=False)
    
    print(f"   [RÉSUMÉ DES SCORES PHYSIQUES]")
    print(f"   -> Final Test Absolu (1000 ep.): {score_total_test:.2f}%")
    print(f"   -> Final Test Bis Normalisé   : {score_total_bis:.2f}% (Fn_norm: {rel_fn_norm:.2f}%, Ft_norm: {rel_ft_norm:.2f}%)")

    if score_total_test < 16.0:
        os.makedirs(f"training/models/{entree}", exist_ok=True)
        torch.save(model_final.state_dict(), f"training/models/{entree}/model_{saved_name}.pth")

def evaluate_baselines(df_test):
    print("\n--- Initialisation de la Baseline ---")
    Fn_s, Ft_s = df_test['Fn_SVEN'].values, df_test['Ft_SVEN'].values
    Fn_b, Ft_b = df_test['Fn_BEM'].values, df_test['Ft_BEM'].values

    rmse_fn = np.sqrt(np.mean((Fn_b - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_b - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    score_total = rel_fn + rel_ft
    wd_score_baseline = wasserstein_distance(Fn_s, Fn_b) + wasserstein_distance(Ft_s, Ft_b)

    # Calcul exact de la Baseline Bis Normalisée (sans unité)
    D = compute_dynamic_pressure_D(df_test)
    Fn_s_norm, Ft_s_norm = Fn_s / D, Ft_s / D
    Fn_b_norm, Ft_b_norm = Fn_b / D, Ft_b / D
    
    rmse_fn_norm = np.sqrt(np.mean((Fn_b_norm - Fn_s_norm)**2))
    rmse_ft_norm = np.sqrt(np.mean((Ft_b_norm - Ft_s_norm)**2))
    rel_fn_norm = (rmse_fn_norm / np.mean(np.abs(Fn_s_norm))) * 100 if np.mean(np.abs(Fn_s_norm)) != 0 else 0
    rel_ft_norm = (rmse_ft_norm / np.mean(np.abs(Ft_s_norm))) * 100 if np.mean(np.abs(Ft_s_norm)) != 0 else 0
    score_total_bis = rel_fn_norm + rel_ft_norm

    results_detail = {
        "Modele": "BASELINE_BEM", "Strat_Entree": "BEM", "Residuelle": "-", "Intermediaire": "-", "Suffixe": "-",
        "RMSE_Fn": round(rmse_fn, 4), "RMSE_Ft": round(rmse_ft, 4), 
        "RMSE_Fn_norm": round(rmse_fn_norm, 4), "RMSE_Ft_norm": round(rmse_ft_norm, 4),
        "Rel_Fn (%)": round(rel_fn, 2), "Rel_Ft (%)": round(rel_ft, 2),
        "Rel_Fn_norm (%)": round(rel_fn_norm, 2), "Rel_Ft_norm (%)": round(rel_ft_norm, 2),
        "Total_Score_Test (%)": round(score_total, 2), 
        "Total_Score_Test_Bis (%)": round(score_total_bis, 2),
        "Total_Score_CV_150 (%)": -1.0, "Total_Score_CV2_1000 (%)": -1.0, "CV2_Variance (%)": -1.0,
        "Wasserstein_Dist": round(wd_score_baseline, 4)
    }
    
    os.makedirs("training/performance", exist_ok=True)
    recap_path = "training/performance/recap_scores_globaux.csv"
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != "BASELINE_BEM"]
        cols_to_drop = ["Total_Score_CV2_Best_Epoch", "Total_Score_CV2_Best_1.5Epoch", "Best_Epoch"]
        df_recap = df_recap.drop(columns=[c for c in cols_to_drop if c in df_recap.columns])
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
        
    df_recap.to_csv(recap_path, index=False)
    print(f"   Baseline BEM enregistrée. Score Absolu : {score_total:.2f}% | Score Bis (Norm) : {score_total_bis:.2f}%")