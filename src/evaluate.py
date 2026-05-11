import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wasserstein_distance
from .models import TurbineMLP
from .data_loader import format_data, get_splits
from .physics import convert_v_to_f


def reconstruct_predictions(df_test, preds, entree, residuelle, inter):
    """Réaligne les prédictions avec r, theta et yaw et gère le résidu BEM."""
    res_str = str(residuelle)
    
    # 1.stratégie f vs v
    if inter == 'f':
        c1, c2 = 'Fn', 'Ft'
        c1_bem, c2_bem = 'Fn_BEM', 'Ft_BEM'
    elif inter == 'v':
        c1, c2 = 'V_eff', 'alpha'
        c1_bem, c2_bem = 'V_eff_BEM', 'alpha_BEM'
        
    records = []
    
    # 2. Reconstitution des prédictions
    if entree == 'L':
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            v1, v2 = preds[i, 0], preds[i, 1]
            if res_str == '1': 
                v1 += row[c1_bem]
                v2 += row[c2_bem]

            records.append({'r': row['r'], 'theta': row['theta'], 'yaw': row['yaw'], f'{c1}_pred': v1, f'{c2}_pred': v2})
            
    elif entree == 'GR':
        # Groupement par theta ET yaw
        for i, (name, group) in enumerate(df_test.groupby(['theta', 'yaw'])):
            group = group.sort_values('r')
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2]
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if res_str == '1': 
                    v1 += row[c1_bem]; v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], 'yaw': row['yaw'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    elif entree == 'GA':
        # Groupement par r ET yaw
        for i, (name, group) in enumerate(df_test.groupby(['r', 'yaw'])):
            group = group.sort_values('theta')
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2]
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if res_str == '1': 
                    v1 += row[c1_bem]; v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], 'yaw': row['yaw'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    elif entree == 'G':
        # Groupement par yaw uniquement
        for i, (y_val, group) in enumerate(df_test.groupby('yaw')):
            group = group.sort_values(['theta', 'r'])
            # preds[i] est le vecteur de taille 2592
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2] 
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if res_str == '1':
                    v1 += row[c1_bem]; v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], 'yaw': row['yaw'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    df_preds = pd.DataFrame(records)
    

    return pd.merge(df_test, df_preds, on=['r', 'theta', 'yaw'])

def evaluator(df_train, df_test, entree, residuelle, inter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"{entree}_{residuelle}_{inter}"
    print(f"--- Évaluation : {model_name} ({device}) ---")
    
    # 1. Hyperparamètres
    hp_path = f"hyperparametres/{model_name}.json"
    if not os.path.exists(hp_path): return
    with open(hp_path, "r") as f: hparams = json.load(f)
        
    # 2. Données (100% pour l'entraînement, pas de split)
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
    
    X_full_train_dev = X_full_train.to(device)
    Y_full_train_dev = Y_full_train.to(device)
    X_test_dev = X_test.to(device)
    
    # 3. Modèle
    model = TurbineMLP(X_full_train.shape[1], Y_full_train.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    if entree == 'G':
        b_size = len(X_full_train_dev)
    elif entree in ['GR', 'GA']:
        b_size = 32
    else:
        b_size = 1024
        
    train_loader = DataLoader(TensorDataset(X_full_train_dev, Y_full_train_dev), batch_size=b_size, shuffle=True)

    # 4. Entraînement Fixe (1000 epochs)
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    
    # 5. Prédictions & Dénormalisation
    model.eval()
    with torch.no_grad(): 
        preds_raw = model(X_test_dev).cpu().numpy()
        
    test_loss_mse = np.mean((preds_raw - Y_test.numpy())**2)
    
    with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f: scaler_Y = pickle.load(f)
    preds_denorm = scaler_Y.inverse_transform(preds_raw)
    
    # 6. Physique
    df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)
    if inter == 'u':
        df_res['V_eff_pred'], df_res['alpha_pred'] = convert_u_to_v(df_res['a_pred'].values, df_res['phi_pred'].values, df_res['r'].values)
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    elif inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    
    # 7. Métriques Finales
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    
    rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    wass_fn = wasserstein_distance(Fn_s, Fn_p)
    wass_ft = wasserstein_distance(Ft_s, Ft_p)

    # 8. DICTIONNAIRES
    results_detail = {
        "Modele": model_name,
        "Epochs_Conv": epochs,
        "Score_Global_%": rel_fn + rel_ft,
        "Loss_Test_MSE": test_loss_mse,
        "RMSE_Fn_Abs": rmse_fn,
        "RMSE_Fn_Rel_%": rel_fn,
        "Wasserstein_Fn": wass_fn,
        "RMSE_Ft_Abs": rmse_ft,
        "RMSE_Ft_Rel_%": rel_ft,
        "Wasserstein_Ft": wass_ft
    }
    
    if inter in ['u', 'v']:
        results_detail["RMSE_Veff_Abs"] = np.sqrt(np.mean((df_res['V_eff_pred'].values - df_res['V_eff_SVEN'].values)**2))
        results_detail["RMSE_Alpha_Abs"] = np.sqrt(np.mean((df_res['alpha_pred'].values - df_res['alpha_SVEN'].values)**2))
    if inter == 'u':
        results_detail["RMSE_Induction_a"] = np.sqrt(np.mean((df_res['a_pred'].values - df_res['a_SVEN'].values)**2))
        results_detail["RMSE_AngleFlux_phi"] = np.sqrt(np.mean((df_res['phi_pred'].values - df_res['phi_SVEN'].values)**2))

    recap_data = {
        "Modele": model_name,
        "Epochs_Conv": epochs,
        "Score_Global_%": rel_fn + rel_ft,
        "RMSE_Fn_Rel_%": rel_fn,
        "RMSE_Ft_Rel_%": rel_ft,
        "Wasserstein_Fn": wass_fn,
        "Wasserstein_Ft": wass_ft
    }

    # 9. SAUVEGARDE
    os.makedirs("performance", exist_ok=True)
    pd.DataFrame([results_detail]).to_csv(f"performance/results_{model_name}.csv", index=False)

    recap_path = "performance/recap_scores_globaux.csv"
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != model_name]
        df_recap = pd.concat([df_recap, pd.DataFrame([recap_data])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([recap_data])
        
    df_recap = df_recap.sort_values(by="Score_Global_%", ascending=True).reset_index(drop=True)
    df_recap.to_csv(recap_path, index=False)
    
    print(f" Terminé ! Score: {recap_data['Score_Global_%']:.2f}%")

def evaluate_baselines(df_full):
    """
    Calcule les métriques de base pour chaque stratégie spatiale (L, GR, GA)
    afin de comparer les modèles physiques sur les mêmes jeux de test que le ML.
    """
    print("--- Évaluation des Baselines par Stratégie ---")
    
    strategies = ['L', 'GR', 'GA']
    baselines = {
        'BEM': ('Fn_BEM', 'Ft_BEM'),
    }

    recap_path = "performance/recap_scores_globaux.csv"
    os.makedirs("performance", exist_ok=True)
    
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
    else:
        df_recap = pd.DataFrame()

    new_rows = []

    for e in strategies:
        # On récupère le jeu de test spécifique à la stratégie spatiale
        _, df_test = get_splits(df_full, entree=e)
        
        Fn_s = df_test['Fn_SVEN'].values
        Ft_s = df_test['Ft_SVEN'].values
        mean_fn_s = np.mean(np.abs(Fn_s))
        mean_ft_s = np.mean(np.abs(Ft_s))

        for b_name, (col_fn, col_ft) in baselines.items():
            full_name = f"Baseline_{b_name}_{e}"
            
            Fn_p = df_test[col_fn].values
            Ft_p = df_test[col_ft].values
            
            rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
            rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
            rel_fn = (rmse_fn / mean_fn_s) * 100 if mean_fn_s != 0 else 0
            rel_ft = (rmse_ft / mean_ft_s) * 100 if mean_ft_s != 0 else 0
            wass_fn = wasserstein_distance(Fn_s, Fn_p)
            wass_ft = wasserstein_distance(Ft_s, Ft_p)

            recap_data = {
                "Modele": full_name,
                "Epochs_Conv": 0,
                "Score_Global_%": rel_fn + rel_ft,
                "RMSE_Fn_Rel_%": rel_fn,
                "RMSE_Ft_Rel_%": rel_ft,
                "Wasserstein_Fn": wass_fn,
                "Wasserstein_Ft": wass_ft
            }
            
            # Nettoyage si la ligne existe déjà
            if not df_recap.empty and "Modele" in df_recap.columns:
                df_recap = df_recap[df_recap["Modele"] != full_name]
            
            new_rows.append(recap_data)

    df_recap = pd.concat([df_recap, pd.DataFrame(new_rows)], ignore_index=True)
    df_recap = df_recap.sort_values(by="Score_Global_%", ascending=True).reset_index(drop=True)
    df_recap.to_csv(recap_path, index=False)
    print("   >> Baselines enregistrées avec succès.")    