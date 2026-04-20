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
    """
    Désaplatit les tenseurs de prédiction pour les réaligner avec (r, theta).
    Gère automatiquement l'ajout du résidu BEM si residuelle == '1'.
    """
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
    
    # --- LOCALE ---
    if entree == 'L':
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            v1, v2 = preds[i, 0], preds[i, 1]
            if residuelle == '1':
                v1 += row[c1_bem]
                v2 += row[c2_bem]
            records.append({'r': row['r'], 'theta': row['theta'], f'{c1}_pred': v1, f'{c2}_pred': v2})
            
    # --- GLOBALE RAYONS FIXÉS ---
    elif entree == 'GR':
        for i, (th, group) in enumerate(df_test.groupby('theta')):
            group = group.sort_values('r')
            p_v1 = preds[i, 0::2] # Indices pairs
            p_v2 = preds[i, 1::2] # Indices impairs
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if residuelle == '1':
                    v1 += row[c1_bem]
                    v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    # --- GLOBALE AZIMUTS FIXÉS ---
    elif entree == 'GA':
        for i, (r_val, group) in enumerate(df_test.groupby('r')):
            group = group.sort_values('theta')
            p_v1 = preds[i, 0::2]
            p_v2 = preds[i, 1::2]
            for j, (_, row) in enumerate(group.iterrows()):
                v1, v2 = p_v1[j], p_v2[j]
                if residuelle == '1':
                    v1 += row[c1_bem]
                    v2 += row[c2_bem]
                records.append({'r': row['r'], 'theta': row['theta'], f'{c1}_pred': v1, f'{c2}_pred': v2})
                
    df_preds = pd.DataFrame(records)
    return pd.merge(df_test, df_preds, on=['r', 'theta'])


def evaluator(df_train, df_test, entree, residuelle, inter):
    model_name = f"{entree}_{residuelle}_{inter}"
    print(f"--- Démarrage Évaluation : {model_name} ---")
    
    # 1. Chargement des hyperparamètres
    hp_path = f"hyperparametres/{model_name}.json"
    if not os.path.exists(hp_path):
        print(f"Hyperparamètres introuvables pour {model_name}. Lancez l'optimisation d'abord.")
        return
        
    with open(hp_path, "r") as f:
        hparams = json.load(f)
        
    # 2. Préparation des données (Création du Scaler Officiel)
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
    
    # Split train/val pour l'Early Stopping
    X_train, X_val, Y_train, Y_val = train_test_split(X_full_train.numpy(), Y_full_train.numpy(), test_size=0.2, random_state=42)
    X_train, Y_train = torch.tensor(X_train), torch.tensor(Y_train)
    X_val, Y_val = torch.tensor(X_val), torch.tensor(Y_val)
    
    # 3. Initialisation du modèle
    model = TurbineMLP(X_train.shape[1], Y_train.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    # 4. Boucle d'entraînement avec Early Stopping
    max_epochs = 2000
    patience = 50
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs_to_converge = 0
    best_weights = None
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            epochs_to_converge = epoch
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"   Early stopping déclenché à l'epoch {epoch} (Meilleur: {epochs_to_converge})")
            break
            
    # Restaurer les meilleurs poids
    model.load_state_dict(best_weights)
    
    # 5. Prédictions sur le Test Set (Normalisées)
    model.eval()
    with torch.no_grad():
        preds_raw = model(X_test).numpy()
    
    # Loss MSE sur les données test (remplace l'ancien RMSE_NN_Raw)
    test_loss_mse = np.mean((preds_raw - Y_test.numpy())**2)
    
    # --- NOUVEAU : DÉNORMALISATION DES PRÉDICTIONS ---
    with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
        
    preds_denorm = scaler_Y.inverse_transform(preds_raw)
    # -------------------------------------------------
    
    # 6. Reconstruire un DataFrame aligné point par point (avec les valeurs réelles)
    df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)
    
    # 7. Conversions Physiques Vectorisées
    if inter == 'u':
        df_res['V_eff_pred'], df_res['alpha_pred'] = convert_u_to_v(df_res['a_pred'].values, df_res['phi_pred'].values, df_res['r'].values)
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    elif inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    
    # 8. Calcul des Métriques sur les Forces Finales
    Fn_sven = df_res['Fn_SVEN'].values
    Ft_sven = df_res['Ft_SVEN'].values
    Fn_pred = df_res['Fn_pred'].values
    Ft_pred = df_res['Ft_pred'].values
    
    rmse_fn = np.sqrt(np.mean((Fn_pred - Fn_sven)**2))
    rmse_ft = np.sqrt(np.mean((Ft_pred - Ft_sven)**2))
    
    rel_rmse_fn = (rmse_fn / np.mean(np.abs(Fn_sven))) * 100 if np.mean(np.abs(Fn_sven)) != 0 else np.nan
    rel_rmse_ft = (rmse_ft / np.mean(np.abs(Ft_sven))) * 100 if np.mean(np.abs(Ft_sven)) != 0 else np.nan
    
    wass_fn = wasserstein_distance(Fn_sven, Fn_pred)
    wass_ft = wasserstein_distance(Ft_sven, Ft_pred)

    # Dictionnaire de résultats dynamiques
    results = {
        "Modele": model_name,
        "Epochs_Conv": epochs_to_converge,
        "Loss_Test_MSE": test_loss_mse,
        "RMSE_Fn_Abs": rmse_fn,
        "RMSE_Fn_Rel_%": rel_rmse_fn,
        "Wasserstein_Fn": wass_fn,
        "RMSE_Ft_Abs": rmse_ft,
        "RMSE_Ft_Rel_%": rel_rmse_ft,
        "Wasserstein_Ft": wass_ft
    }
    
    # Ajout des intermédiaires si pertinent
    if inter in ['u', 'v']:
        rmse_veff = np.sqrt(np.mean((df_res['V_eff_pred'].values - df_res['V_eff_SVEN'].values)**2))
        rmse_alpha = np.sqrt(np.mean((df_res['alpha_pred'].values - df_res['alpha_SVEN'].values)**2))
        results["RMSE_Veff_Abs"] = rmse_veff
        results["RMSE_Alpha_Abs"] = rmse_alpha
    if inter == 'u':
        rmse_a = np.sqrt(np.mean((df_res['a_pred'].values - df_res['a_SVEN'].values)**2))
        rmse_phi = np.sqrt(np.mean((df_res['phi_pred'].values - df_res['phi_SVEN'].values)**2))
        results["RMSE_Induction_a"] = rmse_a
        results["RMSE_AngleFlux_phi"] = rmse_phi

    # 9. Enregistrement
    os.makedirs("performance", exist_ok=True)
    df_metrics = pd.DataFrame([results])
    df_metrics.to_csv(f"performance/results_{model_name}.csv", index=False)
    
    print(f" Évaluation terminée ! RMSE Fn: {rmse_fn:.2f} [N/m] | Relatif: {rel_rmse_fn:.1f}%")