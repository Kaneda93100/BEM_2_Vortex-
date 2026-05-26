import os
import json
import pickle
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from .models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder
from .data_loader import format_data
from .physics import convert_v_to_f
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prise en compte rigoureuse du suffixe
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    recap_path = "performance/recap_scores_globaux.csv"
    
    print(f"\n{'='*50}")
    print(f" RÉ-ENTRAÎNEMENT FINAL & ÉVALUATION : {saved_name}")
    print(f"{'='*50}")
    
    # 1. Chargement des hyperparamètres globaux
    hp_path = f"hyperparametres/{saved_name}.json"
    if not os.path.exists(hp_path):
        print(f"   [ERREUR] Aucun hyperparamètre trouvé pour {saved_name} à l'adresse {hp_path}")
        return
    with open(hp_path, "r") as f: 
        best_params = json.load(f)
        
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=False, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
    criterion = nn.MSELoss()
    
    # 2. Gestion de l'Auto-encodeur Pré-Calculé
    use_ae = best_params.get('use_autoencoder', False) and suffixe != 'D0'
    
    if use_ae:
        latent_dim = best_params['latent_dim']
        ae_params_path = f"hyperparametres/ae_{saved_name}_params.json"
        ae_weights_path = f"models/ae_{saved_name}.pth"
        
        with open(ae_params_path, "r") as f:
            ae_config = json.load(f)
            
        if entree == 'GM':
            ae_model = ConvolutionalAutoencoder(
                in_channels=Y_train.shape[1], latent_dim=latent_dim, 
                depth=ae_config['ae_depth'], base_filters=ae_config['ae_base_filters'], device=device
            ).to(device)
        else:
            ae_model = LinearAutoencoder(in_features=Y_train.shape[1], latent_dim=latent_dim, device=device).to(device)
            
        ae_model.load_state_dict(torch.load(ae_weights_path, map_location=device))
        ae_model.eval()
        
        with torch.no_grad():
            Y_train_target = ae_model.encode(Y_train)
            Y_test_target = ae_model.encode(Y_test)
        print(f"   [OK] AE rechargé depuis {ae_weights_path}")
    else:
        Y_train_target = Y_train
        Y_test_target = Y_test
        latent_dim = 0

    # 3. Création du set de validation interne (80/20) pour le Early Stopping / Scheduler
    X_tr_np, X_val_np, Y_tr_np, Y_val_np = train_test_split(
        X_train.cpu().numpy(), Y_train_target.cpu().numpy(), test_size=0.2, random_state=42
    )
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    Y_tr = torch.tensor(Y_tr_np, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val_np, dtype=torch.float32).to(device)

    # 4. Instanciation du modèle prédictif principal
    if entree == 'GV':
        current_out = latent_dim if use_ae else Y_train.shape[1]
        model = TurbineMLP(X_train.shape[1], current_out, 
                           best_params['n_layers'], best_params['n_neurons'], 
                           best_params['dropout_rate'], device=device).to(device)
    elif entree == 'GM':
        target_dim = latent_dim if use_ae else Y_train.shape[1]
        model = TurbineCNN(in_channels=X_train.shape[1], out_channels=target_dim, 
                           use_autoencoder=use_ae, latent_dim=latent_dim,
                           n_layers=best_params['n_layers'], base_filters=best_params['base_filters'], 
                           dropout_rate=best_params['dropout_rate'], device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
    
    # 5. Entraînement Final (1000 époques complètes)
    epochs = 1000
    best_val_loss = float('inf')
    best_model_weights = None
    
    pbar = tqdm(range(epochs), desc=f"Training {saved_name}")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), Y_tr)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
             
        if (epoch + 1) % 50 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}", "Val": f"{val_loss:.6f}"})
            
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 6. Inférence et Décompression Réseau
    model.eval()
    with torch.no_grad(): 
        preds_raw = model(X_test)
        if use_ae:
            preds_norm_out = ae_model.decode(preds_raw)
        else:
            preds_norm_out = preds_raw


    preds_norm_np = preds_norm_out.cpu().numpy()
    if entree == 'GM':
        preds_flat = preds_norm_np.reshape(preds_norm_np.shape[0], -1)
    else:
        preds_flat = preds_norm_np

    # 7. Dénormalisation
    with open(f"scalers/scaler_Y_{base_model_name}.pkl", 'rb') as f: 
        scaler_Y = pickle.load(f)
    preds_denorm = scaler_Y.inverse_transform(preds_flat)
    
    # 8. Post-processing Physique et Enregistrement du Score
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
    score_total = rel_fn + rel_ft
    wd_score = wasserstein_distance(Fn_s, Fn_p) + wasserstein_distance(Ft_s, Ft_p)
    
    results_detail = {
        "Modele": saved_name, "Strat_Entree": entree, "Residuelle": residuelle, "Intermediaire": inter, "Suffixe": suffixe,
        "RMSE_Fn": round(rmse_fn, 4), "RMSE_Ft": round(rmse_ft, 4), "Rel_Fn (%)": round(rel_fn, 2), "Rel_Ft (%)": round(rel_ft, 2),
        "Total_Score (%)": round(score_total, 2), "Wasserstein_Dist": round(wd_score, 4)
    }
    
    os.makedirs("performance", exist_ok=True)
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != saved_name]
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
    df_recap.to_csv(recap_path, index=False)
    
    print(f"   Score ajouté au récapitulatif global. Erreur Totale: {score_total:.2f}%")

    # SAUVEGARDE CONDITIONNELLE EN PTH SI LE SCORE EST < 16%
    if score_total < 16.0:
        model_save_path = f"models/model_{saved_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"   [SAUVEGARDE] Performance excellente ({score_total:.2f}% < 16%). Modèle enregistré dans {model_save_path}")
    else:
        print(f"   [INFO] Score de {score_total:.2f}% >= 16%. Le fichier .pth final n'a pas été conservé.")


def evaluate_baselines(df_test):
    """ Initialise le fichier récapitulatif avec la baseline BEM. """
    print("\n--- Initialisation de la Baseline ---")
    Fn_s, Ft_s = df_test['Fn_SVEN'].values, df_test['Ft_SVEN'].values
    Fn_b, Ft_b = df_test['Fn_BEM'].values, df_test['Ft_BEM'].values

    rmse_fn = np.sqrt(np.mean((Fn_b - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_b - Ft_s)**2))
    
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    score_total = rel_fn + rel_ft

    results_detail = {
        "Modele": "BASELINE_BEM", "Strat_Entree": "BEM", "Residuelle": "-", "Intermediaire": "-", "Suffixe": "-",
        "RMSE_Fn": round(rmse_fn, 4), "RMSE_Ft": round(rmse_ft, 4), "Rel_Fn (%)": round(rel_fn, 2), "Rel_Ft (%)": round(rel_ft, 2),
        "Total_Score (%)": round(score_total, 2), "Wasserstein_Dist": -1.0
    }
    
    os.makedirs("performance", exist_ok=True)
    recap_path = "performance/recap_scores_globaux.csv"
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != "BASELINE_BEM"]
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
    df_recap.to_csv(recap_path, index=False)
    print(f"   Baseline BEM enregistrée. Score : {score_total:.2f}%")