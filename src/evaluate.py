import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wasserstein_distance
from .models import TurbineMLP, EnsemblePointNet
from .data_loader import format_data, get_splits
from .physics import convert_v_to_f


def reconstruct_predictions(df_test, preds, entree, residuelle, inter):
    """
    Réaligne les prédictions avec r, theta et yaw et gère le résidu BEM.
    Pour la stratégie 'P', un tri est appliqué pour correspondre à l'ordre de format_data.
    """
    res_str = str(residuelle)
    
    if inter == 'f':
        c1, c2 = 'Fn', 'Ft'
        c1_bem, c2_bem = 'Fn_BEM', 'Ft_BEM'
    elif inter == 'v':
        c1, c2 = 'V_eff', 'alpha'
        c1_bem, c2_bem = 'V_eff_BEM', 'alpha_BEM'
        
    records = []
    
    if entree == 'P':
        # Tri identique à celui utilisé dans data_loader.format_data pour garantir l'alignement
        df_test_sorted = df_test.sort_values(['r', 'theta', 'yaw']).reset_index(drop=True)
        for i in range(len(df_test_sorted)):
            row = df_test_sorted.iloc[i]
            v1, v2 = preds[i, 0], preds[i, 1]
            if res_str == '1': 
                v1 += row[c1_bem]
                v2 += row[c2_bem]
            records.append({
                'r': row['r'], 
                'theta': row['theta'], 
                'yaw': row['yaw'], 
                f'{c1}_pred': v1, 
                f'{c2}_pred': v2
            })
            
    elif entree == 'G':
        # Groupement par yaw uniquement
        for i, (y_val, group) in enumerate(df_test.groupby('yaw')):
            group = group.sort_values(['theta', 'r'])
            # preds[i] est le vecteur de taille 2592 (36x36x2)
            p_v1, p_v2 = preds[i, 0::2], preds[i, 1::2] 
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if res_str == '1':
                    v1 += row[c1_bem]
                    v2 += row[c2_bem]
                records.append({
                    'r': row['r'], 
                    'theta': row['theta'], 
                    'yaw': row['yaw'], 
                    f'{c1}_pred': v1, 
                    f'{c2}_pred': v2
                })
                
    df_preds = pd.DataFrame(records)
    return pd.merge(df_test, df_preds, on=['r', 'theta', 'yaw'])


def evaluator(df_train, df_test, entree, residuelle, inter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"{entree}_{residuelle}_{inter}"
    print(f"--- Évaluation : {model_name} ({device}) ---")
    
    # 1. Chargement des Meilleurs Hyperparamètres
    hp_path = f"hyperparametres/{model_name}.json"
    if not os.path.exists(hp_path):
        print(f"Erreur: Aucun hyperparamètre trouvé pour {model_name}")
        return
    with open(hp_path, "r") as f: 
        hparams = json.load(f)
        
    # 2. Préparation des Données
    # format_data gère le scaling et le stockage des scalers
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
    
    # --- Logique Spécifique pour l'Ensemble de modèles (Stratégie P) ---
    if entree == 'P':
        num_points = 1296
        num_yaws_train = len(X_full_train) // num_points
        num_yaws_test = len(X_test) // num_points
        
        # Passage en vue 3D : (1296 modèles, N_yaws, Features)
        X_train_3D = X_full_train.view(num_points, num_yaws_train, -1)
        Y_train_3D = Y_full_train.view(num_points, num_yaws_train, -1)
        X_test_3D = X_test.view(num_points, num_yaws_test, -1)
        
        model = EnsemblePointNet(
            num_points, X_train_3D.shape[2], Y_train_3D.shape[2], 
            hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']
        ).to(device)
        
        X_train_final, Y_train_final = X_train_3D, Y_train_3D
    else:
        # Stratégie G
        model = TurbineMLP(
            X_full_train.shape[1], Y_full_train.shape[1], 
            hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']
        ).to(device)
        X_train_final, Y_train_final = X_full_train, Y_full_train

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    # 3. Entraînement Final 
    epochs = 1000
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train_final), Y_train_final)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
    
    # 4. Inférence
    model.eval()
    with torch.no_grad(): 
        if entree == 'P':
            # On prédit en 3D puis on "aplatit" pour la dénormalisation
            preds_raw = model(X_test_3D).view(-1, Y_train_final.shape[2]).cpu().numpy()
        else:
            preds_raw = model(X_test).cpu().numpy()
            
    test_loss_mse = np.mean((preds_raw - Y_test.cpu().numpy())**2)
    
    # 5. Dénormalisation
    with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f: 
        scaler_Y = pickle.load(f)
    preds_denorm = scaler_Y.inverse_transform(preds_raw)
    
    # 6. Post-traitement Physique
    df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)
    
    if inter == 'v':
        # Reconstitution des forces Fn, Ft à partir des vitesses prédites
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(
            df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values
        )
    
    # 7. Calcul des Métriques
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    
    rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    
    # 8. Sauvegarde des résultats
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
    pd.DataFrame([results_detail]).to_csv(f"performance/results_{model_name}.csv", index=False)
    print(f"   Terminé ! Erreur Relative Totale: {rel_fn + rel_ft:.2f}%")


def evaluate_baselines(df_test):
    """
    Évaluation de la baseline BEM. 
    """
    print("\n--- Initialisation de la Baseline (Run Vierge) ---")
    
    # Calcul des métriques sur le split fourni (identique à P et G)
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
        "RMSE_Fn_Rel_%": rel_fn,
        "RMSE_Ft_Rel_%": rel_ft,
        "Wass_Fn": wasserstein_distance(Fn_s, Fn_b),
        "Wass_Ft": wasserstein_distance(Ft_s, Ft_b)
    }
    os.makedirs("performance", exist_ok=True)
    pd.DataFrame([recap_data]).to_csv("performance/recap_scores_globaux.csv", index=False)
    
    print(f"   Dossier 'performance/' initialisé avec la Baseline. Score: {rel_fn + rel_ft:.2f}%")