import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from .models import TurbineMLP
from .data_loader import format_data
from .physics import convert_u_to_v, convert_v_to_f

def reconstruct_predictions(df_test, preds, entree, residuelle, inter):
    """Réaligne les prédictions avec r et theta et gère le résidu BEM."""
    if inter == 'f':
        c1, c2 = 'Fn', 'Ft'
        c1_bem, c2_bem = 'Fn_BEM', 'Ft_BEM'
    elif inter == 'v':
        c1, c2 = 'V_eff', 'alpha'
        c1_bem, c2_bem = 'V_eff_BEM', 'alpha_BEM'
    elif inter == 'u':
        c1, c2 = 'a', 'phi'
        c1_bem, c2_bem = 'a_BEM', 'phi_BEM'
        
    records = []
    
    if entree == 'L':
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            v1, v2 = preds[i, 0], preds[i, 1]
            if residuelle == '1':
                v1 += row[c1_bem]
                v2 += row[c2_bem]
            records.append({'r': row['r'], 'theta': row['theta'], f'{c1}_pred': v1, f'{c2}_pred': v2})
            
    elif entree == 'GR':
        for i, (th, group) in enumerate(df_test.groupby('theta')):
            group = group.sort_values('r')
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2]
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if residuelle == '1':
                    v1 += row[c1_bem]; v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    elif entree == 'GA':
        for i, (r_val, group) in enumerate(df_test.groupby('r')):
            group = group.sort_values('theta')
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2]
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if residuelle == '1':
                    v1 += row[c1_bem]; v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    df_preds = pd.DataFrame(records)
    return pd.merge(df_test, df_preds, on=['r', 'theta'])

def evaluator(df_train, df_test, entree, residuelle, inter):
    # Support GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"{entree}_{residuelle}_{inter}"
    print(f"--- Évaluation : {model_name} ({device}) ---")
    
    # 1. Hyperparamètres
    hp_path = f"hyperparametres/{model_name}.json"
    if not os.path.exists(hp_path): return
    with open(hp_path, "r") as f: hparams = json.load(f)
        
    # 2. Données
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
    
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_full_train.numpy(), Y_full_train.numpy(), test_size=0.2, random_state=42)
    
    X_tr, Y_tr = torch.tensor(X_tr).to(device), torch.tensor(Y_tr).to(device)
    X_val, Y_val = torch.tensor(X_val).to(device), torch.tensor(Y_val).to(device)
    X_test_dev = X_test.to(device)
    
    # 3. Modèle
    model = TurbineMLP(X_tr.shape[1], Y_tr.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    # 4. Entraînement
    best_val_loss, epochs_no_improve, epochs_to_converge, best_weights = float('inf'), 0, 0, None
    for epoch in range(2000):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_tr), Y_tr)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss; epochs_no_improve = 0; epochs_to_converge = epoch; best_weights = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 50: break
            
    model.load_state_dict(best_weights)
    
    # 5. Prédictions & Dénormalisation
    model.eval()
    with torch.no_grad(): preds_raw = model(X_test_dev).cpu().numpy()
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

    # 8. RECAPITULATIF 
    recap_data = {
        "Modele": model_name,
        "Epochs_Conv": epochs_to_converge,
        "Score_Global_%": rel_fn + rel_ft,
        "RMSE_Fn_Rel_%": rel_fn,
        "RMSE_Ft_Rel_%": rel_ft,
        "Wasserstein_Fn": wass_fn,
        "Wasserstein_Ft": wass_ft
    }

    # Sauvegarde globale
    os.makedirs("performance", exist_ok=True)
    recap_path = "performance/recap_scores_globaux.csv"
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != model_name]
        df_recap = pd.concat([df_recap, pd.DataFrame([recap_data])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([recap_data])
        
    df_recap = df_recap.sort_values(by="Score_Global_%", ascending=True).reset_index(drop=True)
    df_recap.to_csv(recap_path, index=False)
    
    print(f"   Terminé ! Score: {recap_data['Score_Global_%']:.2f}%")