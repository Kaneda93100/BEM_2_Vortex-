import os
import json
import pickle
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import KFold
from core.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, DecoderLoss, PhysicsInformedLoss, TorchScaler, convert_v_to_f_torch
from .data_loader import format_data
from core.physics import convert_v_to_f, get_geometry
from tqdm import tqdm

def reconstruct_predictions(df_test, preds, entree, residuelle, inter):
    """ Réaligne les prédictions (déjà décodées et dénormalisées) selon la topologie d'origine. """
    res_str = str(residuelle)
    if inter == 'f':
        c1, c2 = 'Fn', 'Ft'
        c1_bem, c2_bem = 'Fn_BEM', 'Ft_BEM'
    elif inter == 'v':
        c1, c2 = 'V_eff', 'alpha'
        c1_bem, c2_bem = 'V_eff_BEM', 'alpha_BEM'
        
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
    V_BEM_phys_test = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    recap_path = "training/performance/recap_scores_globaux.csv"
    is_cnn = (entree == 'GM')
    
    print(f"\n{'='*50}")
    print(f" ÉVALUATION EXHAUSTIVE (3 Folds CV | Fixe 1000 Epochs) : {saved_name}")
    print(f"{'='*50}")
    
    hp_path = f"training/hyperparametres/{entree.lower()}_hyperparameters.json"
    if not os.path.exists(hp_path): return
        
    with open(hp_path, "r") as f: all_hps = json.load(f)
    if saved_name not in all_hps: return
    best_params = all_hps[saved_name]
        
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=False, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)

    # =========================================================================
    # INITIALISATION : SCALERS & RÉFÉRENCES PHYSIQUES
    # =========================================================================
    # ON CHARGE LES VRAIES FORCES SVEN (ABS) POUR TOUT LE MONDE (Dénominateur de l'erreur)
    _, Y_f_abs_tr = format_data(df_train, entree, '0', 'f', is_train=False, device=device)
    _, Y_f_abs_te = format_data(df_test, entree, '0', 'f', is_train=False, device=device)
    with open(f"training/scalers/scaler_Y_{entree}_0_f.pkl", 'rb') as f: scaler_f_abs = pickle.load(f)
    scaler_f_abs_torch = TorchScaler(scaler_f_abs, device)
    
    F_SVEN_phys_train_abs = scaler_f_abs_torch.inverse_transform(Y_f_abs_tr)
    F_SVEN_phys_test_abs = scaler_f_abs_torch.inverse_transform(Y_f_abs_te)

    if inter == 'v':
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
            c_grid = np.array([geom.get_chord(r) for r in r_uniques])
            C_grid, _ = np.meshgrid(c_grid, theta_uniques, indexing='ij')
            r_tensor = torch.tensor(R_grid, dtype=torch.float32, device=device)
            c_tensor = torch.tensor(C_grid, dtype=torch.float32, device=device)
        else:
            group = df_train[(df_train['yaw'] == df_train['yaw'].iloc[0])]
            if 'TSR' in group.columns: group = group[group['TSR'] == group['TSR'].iloc[0]]
            group = group.sort_values(['theta', 'r'])
            r_array = group['r'].values
            r_tensor = torch.tensor(r_array, dtype=torch.float32, device=device)
            c_tensor = torch.tensor(np.array([geom.get_chord(r) for r in r_array]), dtype=torch.float32, device=device)

        if str(residuelle) == '1':
            _, Y_train_abs_v = format_data(df_train, entree, '0', inter, is_train=False, device=device)
            _, Y_test_abs_v = format_data(df_test, entree, '0', inter, is_train=False, device=device)
            with open(f"training/scalers/scaler_Y_{entree}_0_v.pkl", 'rb') as f_abs_v: scaler_v_abs = pickle.load(f_abs_v)
            scaler_v_abs_torch = TorchScaler(scaler_v_abs, device)
            
            V_SVEN_phys_tr = scaler_v_abs_torch.inverse_transform(Y_train_abs_v)
            Delta_V_phys_tr = scaler_v_torch.inverse_transform(Y_train)
            V_BEM_phys_train = V_SVEN_phys_tr - Delta_V_phys_tr
            
            V_SVEN_phys_te = scaler_v_abs_torch.inverse_transform(Y_test_abs_v)
            Delta_V_phys_te = scaler_v_torch.inverse_transform(Y_test)
            V_BEM_phys_test = V_SVEN_phys_te - Delta_V_phys_te
            
        F_TARGET_phys_train = F_SVEN_phys_train_abs 
        F_TARGET_phys_test = F_SVEN_phys_test_abs
    else:
        with open(f"training/scalers/scaler_Y_{entree}_{residuelle}_f.pkl", 'rb') as f: scaler_f = pickle.load(f)
        scaler_f_torch = TorchScaler(scaler_f, device)
        F_TARGET_phys_train = scaler_f_torch.inverse_transform(Y_train)
        F_TARGET_phys_test = scaler_f_torch.inverse_transform(Y_test)

    use_ae = best_params.get('use_autoencoder', False) and suffixe != 'D0'
    Y_train_target = Y_train
    
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

    if inter == 'v':
        criterion = PhysicsInformedLoss(current_ae, scaler_v, scaler_f, 0.5, r_tensor, c_tensor, polar_surrogate, device)
    else:
        criterion = DecoderLoss(current_ae)

    def compute_metrics(model, X_eval, target_phys_eval, abs_sven_phys_eval, v_bem_phys_eval=None):
        with torch.no_grad():
            preds_norm = current_ae.decode(model(X_eval)) if use_ae else model(X_eval)
            
            if inter == 'v':
                v_pred_phys = scaler_v_torch.inverse_transform(preds_norm)
                if v_bem_phys_eval is not None: v_pred_phys = v_pred_phys + v_bem_phys_eval

                if is_cnn:
                    v_eff_p, alpha_p = v_pred_phys[:, 0], v_pred_phys[:, 1]
                else:
                    v_eff_p, alpha_p = v_pred_phys[:, 0::2], v_pred_phys[:, 1::2]
                    
                f_pred_phys = convert_v_to_f_torch(v_eff_p, alpha_p, r_tensor, c_tensor, polar_surrogate)
                
                if is_cnn:
                    f_pred_phys = f_pred_phys.permute(0, 3, 1, 2)
                    Fn_p, Ft_p = f_pred_phys[:, 0], f_pred_phys[:, 1]
                    Fn_t_abs, Ft_t_abs = abs_sven_phys_eval[:, 0], abs_sven_phys_eval[:, 1]
                else:
                    Fn_p, Ft_p = f_pred_phys[..., 0], f_pred_phys[..., 1]
                    Fn_t_abs, Ft_t_abs = abs_sven_phys_eval[:, 0::2], abs_sven_phys_eval[:, 1::2]
                    
                rmse_fn = torch.sqrt(torch.mean((Fn_p - Fn_t_abs)**2))
                rmse_ft = torch.sqrt(torch.mean((Ft_p - Ft_t_abs)**2))
                mean_fn_eval = torch.mean(torch.abs(Fn_t_abs))
                mean_ft_eval = torch.mean(torch.abs(Ft_t_abs))
            else:
                f_pred_phys = scaler_f_torch.inverse_transform(preds_norm)
                if is_cnn:
                    Fn_p, Ft_p = f_pred_phys[:, 0], f_pred_phys[:, 1]
                    Fn_t_target, Ft_t_target = target_phys_eval[:, 0], target_phys_eval[:, 1]
                    Fn_t_abs, Ft_t_abs = abs_sven_phys_eval[:, 0], abs_sven_phys_eval[:, 1]
                else:
                    Fn_p, Ft_p = f_pred_phys[:, 0::2], f_pred_phys[:, 1::2]
                    Fn_t_target, Ft_t_target = target_phys_eval[:, 0::2], target_phys_eval[:, 1::2]
                    Fn_t_abs, Ft_t_abs = abs_sven_phys_eval[:, 0::2], abs_sven_phys_eval[:, 1::2]
                    
                rmse_fn = torch.sqrt(torch.mean((Fn_p - Fn_t_target)**2))
                rmse_ft = torch.sqrt(torch.mean((Ft_p - Ft_t_target)**2))
                mean_fn_eval = torch.mean(torch.abs(Fn_t_abs))
                mean_ft_eval = torch.mean(torch.abs(Ft_t_abs))
            
            rel_fn = (rmse_fn / mean_fn_eval * 100) if mean_fn_eval > 0 else 0
            rel_ft = (rmse_ft / mean_ft_eval * 100) if mean_ft_eval > 0 else 0
            
            return (rel_fn + rel_ft).item()

    # =========================================================================
    # PHASE 1 : VALIDATION CROISÉE SUR 3 FOLDS (1000 ÉPOQUES)
    # =========================================================================
    print("\n   [1/2] Lancement de la Cross-Validation (3 Folds x 1000 époques)...")
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = np.zeros(n_splits)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train.cpu().numpy())):
        X_tr_cv, Y_tr_cv = X_train[train_idx], Y_train_target[train_idx]
        X_val_cv = X_train[val_idx]
        
        v_bem_tr_cv = V_BEM_phys_train[train_idx] if V_BEM_phys_train is not None else None
        v_bem_val_cv = V_BEM_phys_train[val_idx] if V_BEM_phys_train is not None else None
        
        if entree == 'GV':
            model_cv = TurbineMLP(X_train.shape[1], target_dim, best_params['n_layers'], best_params['n_neurons'], best_params['dropout_rate'], device=device).to(device)
        elif entree == 'GM':
            model_cv = TurbineCNN(in_channels=X_train.shape[1], out_channels=target_dim, use_autoencoder=use_ae, latent_dim=latent_dim,
                               n_layers=best_params['n_layers'], base_filters=best_params['base_filters'], dropout_rate=best_params['dropout_rate'], device=device).to(device)
        
        optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=best_params['lr'])
        
        pbar_cv = tqdm(range(1000), desc=f"   -> Fold {fold+1}/{n_splits}", leave=False)
        for epoch in pbar_cv:
            model_cv.train()
            optimizer_cv.zero_grad()
            
            if inter == 'v' and V_BEM_phys_train is not None:
                loss_cv = criterion(model_cv(X_tr_cv), Y_tr_cv, v_bem_phys=v_bem_tr_cv)
            else:
                loss_cv = criterion(model_cv(X_tr_cv), Y_tr_cv)
                
            loss_cv.backward()
            optimizer_cv.step()
            
        # Évaluation finale du fold après 1000 epochs
        model_cv.eval()
        score_val = compute_metrics(model_cv, X_val_cv, F_TARGET_phys_train[val_idx], F_SVEN_phys_train_abs[val_idx], v_bem_val_cv)
        cv_scores[fold] = score_val

    # --- Calcul de la CV à 1000 Epochs ---
    mean_cv_1000 = cv_scores.mean()
    std_cv_1000 = cv_scores.std()

    # =========================================================================
    # PHASE 2 : ENTRAÎNEMENT DU MODÈLE FINAL SUR 1000 EPOCHS
    # =========================================================================
    print(f"\n   [2/2] Entraînement complet du Modèle Final (100% Data) sur 1000 époques...")
    if entree == 'GV':
        model_final = TurbineMLP(X_train.shape[1], target_dim, best_params['n_layers'], best_params['n_neurons'], best_params['dropout_rate'], device=device).to(device)
    elif entree == 'GM':
        model_final = TurbineCNN(in_channels=X_train.shape[1], out_channels=target_dim, use_autoencoder=use_ae, latent_dim=latent_dim,
                           n_layers=best_params['n_layers'], base_filters=best_params['base_filters'], dropout_rate=best_params['dropout_rate'], device=device).to(device)

    optimizer_final = torch.optim.Adam(model_final.parameters(), lr=best_params['lr'])
    best_train_loss = float('inf')
    best_model_weights = None
    
    pbar = tqdm(range(1000), desc=f"   Training Final")
    for epoch in pbar:
        model_final.train()
        optimizer_final.zero_grad()
        if inter == 'v' and V_BEM_phys_train is not None:
            loss = criterion(model_final(X_train), Y_train_target, v_bem_phys=V_BEM_phys_train)
        else:
            loss = criterion(model_final(X_train), Y_train_target)
        loss.backward()
        optimizer_final.step()
        
        current_loss = loss.item()
        
        if current_loss < best_train_loss:
            best_train_loss = current_loss
            best_model_weights = copy.deepcopy(model_final.state_dict())
             
        if (epoch + 1) % 50 == 0: pbar.set_postfix({"Loss": f"{current_loss:.6f}"})
            
    if best_model_weights is not None:
        model_final.load_state_dict(best_model_weights)

    # --- CALCULS FINAUX SUR LE JEU DE TEST ---
    model_final.eval()
    with torch.no_grad(): 
        preds_raw = model_final(X_test)
        preds_norm_out = current_ae.decode(preds_raw) if use_ae else preds_raw

    preds_norm_np = preds_norm_out.cpu().numpy()
    preds_flat = preds_norm_np.reshape(preds_norm_np.shape[0], -1) if entree == 'GM' else preds_norm_np

    with open(f"training/scalers/scaler_Y_{base_model_name}.pkl", 'rb') as f: scaler_Y = pickle.load(f)
    preds_denorm = scaler_Y.inverse_transform(preds_flat)
    
    df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)
    if inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    
    rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    score_total_test = rel_fn + rel_ft
    wd_score = wasserstein_distance(Fn_s, Fn_p) + wasserstein_distance(Ft_s, Ft_p)
    
    results_detail = {
        "Modele": saved_name, "Strat_Entree": entree, "Residuelle": residuelle, "Intermediaire": inter, "Suffixe": suffixe,
        "RMSE_Fn": round(rmse_fn, 4), "RMSE_Ft": round(rmse_ft, 4), 
        "Rel_Fn (%)": round(rel_fn, 2), "Rel_Ft (%)": round(rel_ft, 2),
        "Total_Score_Test (%)": round(score_total_test, 2),
        "Total_Score_CV_150 (%)": round(best_params.get("Total_Score_CV", -1.0), 2),
        "Total_Score_CV2_1000 (%)": round(mean_cv_1000, 2), 
        "CV2_Variance (%)": round(std_cv_1000, 2),
        "Wasserstein_Dist": round(wd_score, 4)
    }
    
    os.makedirs("performance", exist_ok=True)
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != saved_name]
        # Suppression des colonnes obsolètes si elles sont présentes
        cols_to_drop = ["Total_Score_CV2_Best_Epoch", "Total_Score_CV2_Best_1.5Epoch", "Best_Epoch"]
        df_recap = df_recap.drop(columns=[c for c in cols_to_drop if c in df_recap.columns])
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
        
    df_recap.to_csv(recap_path, index=False)
    
    print(f"   [RÉSUMÉ DES SCORES PHYSIQUES]")
    print(f"   -> Optuna CV (150 ep.) : {best_params.get('Total_Score_CV', -1.0):.2f}%")
    print(f"   -> Final CV  (1000 ep.): {mean_cv_1000:.2f}% (Var: {std_cv_1000:.2f}%)")
    print(f"   -> Final Test(1000 ep.): {score_total_test:.2f}%")

    if score_total_test < 16.0:
        os.makedirs(f"training/models/{entree}", exist_ok=True)
        model_save_path = f"training/models/{entree}/model_{saved_name}.pth"
        torch.save(model_final.state_dict(), model_save_path)
        print(f"   [SAUVEGARDE] Performance excellente (<16%). Modèle enregistré dans {model_save_path}")

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

    results_detail = {
        "Modele": "BASELINE_BEM", "Strat_Entree": "BEM", "Residuelle": "-", "Intermediaire": "-", "Suffixe": "-",
        "RMSE_Fn": round(rmse_fn, 4), "RMSE_Ft": round(rmse_ft, 4), 
        "Rel_Fn (%)": round(rel_fn, 2), "Rel_Ft (%)": round(rel_ft, 2),
        "Total_Score_Test (%)": round(score_total, 2), 
        "Total_Score_CV_150 (%)": -1.0, 
        "Total_Score_CV2_1000 (%)": -1.0, 
        "CV2_Variance (%)": -1.0,
        "Wasserstein_Dist": round(wd_score_baseline, 4)
    }
    
    os.makedirs("performance", exist_ok=True)
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
    print(f"   Baseline BEM enregistrée. Score : {score_total:.2f}% | WD : {wd_score_baseline:.4f}")