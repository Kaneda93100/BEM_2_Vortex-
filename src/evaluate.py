import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from .models import TurbineMLP, TurbineCNN
from .data_loader import format_data
from .physics import convert_v_to_f
from tqdm import tqdm

def reconstruct_predictions(df_test, preds, entree, residuelle, inter):
    """ Réaligne les prédictions selon la topologie d'origine (GV ou GM). """
    res_str = str(residuelle)
    if inter == 'f':
        c1, c2 = 'Fn', 'Ft'
        c1_bem, c2_bem = 'Fn_BEM', 'Ft_BEM'
    elif inter == 'v':
        c1, c2 = 'V_eff', 'alpha'
        c1_bem, c2_bem = 'V_eff_BEM', 'alpha_BEM'
        
    records = []
    
    # On itère sur chaque Yaw du jeu de test
    for i, (y_val, group) in enumerate(df_test.groupby('yaw')):
        if entree == 'GV':
            # GV trié sur [theta, r] dans data_loader
            group = group.sort_values(['theta', 'r'])
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2]
            
        elif entree == 'GM':
            # GM trié sur [r, theta] pour l'image
            group = group.sort_values(['r', 'theta'])
            num_r = len(group['r'].unique())
            num_theta = len(group['theta'].unique())
            # On reforme l'image (2, 36, 36)
            pred_img = preds[i].reshape(2, num_r, num_theta)
            p_v1, p_v2 = pred_img[0].flatten(), pred_img[1].flatten()
            
        for j, (_, row) in enumerate(group.iterrows()):
            v1, v2 = p_v1[j], p_v2[j]
            if res_str == '1':
                v1 += row[c1_bem]
                v2 += row[c2_bem]
            records.append({
                'r': row['r'], 'theta': row['theta'], 'yaw': row['yaw'], 
                f'{c1}_pred': v1, f'{c2}_pred': v2
            })
                
    df_preds = pd.DataFrame(records)
    return pd.merge(df_test, df_preds, on=['r', 'theta', 'yaw'])


def evaluator(df_train, df_test, entree, residuelle, inter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"{entree}_{residuelle}_{inter}"
    recap_path = "performance/recap_scores_globaux.csv"
    
    print(f"--- Évaluation : {model_name} ({device}) ---")
    
    # 1. Chargement hyperparamètres
    hp_path = f"hyperparametres/{model_name}.json"
    if not os.path.exists(hp_path):
        print(f"Erreur: Aucun hyperparamètre trouvé pour {model_name}")
        return
    with open(hp_path, "r") as f: 
        hparams = json.load(f)
        
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
    
    # 2. Instanciation du modèle
    if entree == 'GV':
        model = TurbineMLP(X_train.shape[1], Y_train.shape[1], 
                           hparams['n_layers'], hparams['n_neurons'], 
                           hparams['dropout_rate'], device=device).to(device)
    elif entree == 'GM':
        model = TurbineCNN(in_channels=3, out_channels=2, 
                           n_layers=hparams['n_layers'], base_filters=hparams['base_filters'], 
                           dropout_rate=hparams['dropout_rate'], device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    # 3. Entraînement Final (Full Batch)
    epochs = 1000
    model.train()
    pbar = tqdm(range(epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
    
    # 4. Inférence
    model.eval()
    with torch.no_grad(): 
        preds_raw = model(X_test).cpu().numpy()
        
    test_loss_mse = np.mean((preds_raw - Y_test.cpu().numpy())**2)
    
    if entree == 'GM':
        preds_raw = preds_raw.reshape(preds_raw.shape[0], -1)

    with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f: 
        scaler_Y = pickle.load(f)
    preds_denorm = scaler_Y.inverse_transform(preds_raw)
    
    # 5. Reconstruction et Métriques
    df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)
    
    if inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(
            df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values
        )
    
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    
    rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    
    # 6. SAUVEGARDE DANS LE RÉCAPITULATIF GLOBAL
    results_detail = {
        "Modele": model_name,
        "Epochs": epochs,
        "Score_Total_%": rel_fn + rel_ft,
        "Loss_MSE_Test": test_loss_mse,
        "RMSE_Fn_Rel_%": rel_fn,
        "RMSE_Ft_Rel_%": rel_ft,
        "Wass_Fn": wasserstein_distance(Fn_s, Fn_p),
        "Wass_Ft": wasserstein_distance(Ft_s, Ft_p)
    }
    
    os.makedirs("performance", exist_ok=True)
    
    # On charge le fichier existant (qui contient au moins la baseline)
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        # Supprime l'ancienne entrée de ce modèle si elle existe (pour éviter les doublons)
        df_recap = df_recap[df_recap["Modele"] != model_name]
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
        
    df_recap.to_csv(recap_path, index=False)
    print(f"   Score ajouté au récapitulatif global. Erreur Totale: {rel_fn + rel_ft:.2f}%")


def evaluate_baselines(df_test):
    """ Initialise le fichier récapitulatif avec la baseline BEM. """
    print("\n--- Initialisation de la Baseline ---")
    
    Fn_s, Ft_s = df_test['Fn_SVEN'].values, df_test['Ft_SVEN'].values
    Fn_b, Ft_b = df_test['Fn_BEM'].values, df_test['Ft_BEM'].values

    rmse_fn = np.sqrt(np.mean((Fn_b - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_b - Ft_s)**2))
    
    mean_fn_s = np.mean(np.abs(Fn_s))
    mean_ft_s = np.mean(np.abs(Ft_s))
    
    rel_fn = (rmse_fn / mean_fn_s) * 100 if mean_fn_s != 0 else 0
    rel_ft = (rmse_ft / mean_ft_s) * 100 if mean_ft_s != 0 else 0
    
    recap_data = {
        "Modele": "Baseline_BEM_Pure",
        "Epochs": 0,
        "Score_Total_%": rel_fn + rel_ft,
        "Loss_MSE_Test": 0.0,
        "RMSE_Fn_Rel_%": rel_fn,
        "RMSE_Ft_Rel_%": rel_ft,
        "Wass_Fn": wasserstein_distance(Fn_s, Fn_b),
        "Wass_Ft": wasserstein_distance(Ft_s, Ft_b)
    }
    
    os.makedirs("performance", exist_ok=True)
    pd.DataFrame([recap_data]).to_csv("performance/recap_scores_globaux.csv", index=False)
    print(f"   Baseline BEM enregistrée. Score: {rel_fn + rel_ft:.2f}%")