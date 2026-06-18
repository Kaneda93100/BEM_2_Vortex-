import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

import json
import pickle
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

from Power_models.src.dataloaders import format_data_power
from Power_models.src.PModels import PowerMLP
from core.models import TorchScaler
from core.physics import compute_cp, convert_v_to_f
from Power_models.src.dataloaders import convert_f_to_power

def evaluator_power(df_train, df_val, entree, res, comp) :
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = f'{entree}_{res}'
    recap_path = 'Power_models/performances/recap_score.csv'
    saved_name = model_name

    print(f"\n{'='*50}")
    print(f" ÉVALUATION EXHAUSTIVE (3 Folds CV | Fixe 1000 Epochs) : {saved_name}")
    print(f"{'='*50}")

    hp_path = f"Power_models/HP/{entree}_hp.json"
    if not os.path.exists(hp_path) :
        print("\nErreur, les hyperparamètres de l'optimisation n'ont pas été trouvé.\n")
        return
    
    with open(hp_path, "r") as f: all_hps = json.load(f)
    if saved_name not in all_hps: return
    best_params = all_hps[saved_name]

    X_train, Y_train = format_data_power(df_train, entree, res, comp, device = device)
    X_val, Y_val     = format_data_power(df_val, entree, res, comp, device = device)


    ## Récupérer les puissances de références pour la Cross-Validation
    _, true_tr = format_data_power(df_train, entree, res = '0', comp = False, scaler_exist = False, device = device)
    _, pow_sven_val = format_data_power(df_val, entree, res = '0', comp = False, scaler_exist = False, device = device) 
    
    with open(f'Power_models/Data/DataDir/scalers/scaler_Y_{entree}_0_pow.pkl', 'rb') as f :
        scaler_Y = pickle.load(f)
    scaler_TPow_torch = TorchScaler(scaler_Y, device = device)

    model = PowerMLP(X_train.shape[1], Y_train.shape[1], 
                          n_layers = best_params['n_layers'], n_neurons = best_params['n_neurons'],
                          dropout = best_params['dropout_rate'], device=device)

    print("\n   [1/2] Lancement de la Cross-Validation (3 Folds x 1000 époques)...")
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = np.zeros(n_splits)
    crit = torch.nn.MSELoss()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train.cpu().numpy())) :
        X_tr_cv, Y_tr_cv    = X_train[train_idx], Y_train[train_idx]
        X_val_cv, Y_val_cv  = X_train[val_idx], Y_train[val_idx]

        optimizer = torch.optim.Adam(model.parameters(), lr = best_params['lr'])
        pbar = tqdm(range(10), desc=f"   -> Fold {fold+1}/{n_splits}", leave=False)

        for epoch in pbar :
            model.train()
            optimizer.zero_grad()

            loss = crit(model(X_tr_cv), Y_tr_cv)
            loss.backward()
            optimizer.step()

        model.eval()
        pred_raw = model(X_val_cv)
        pred_denorm = scaler_TPow_torch.inverse_transform(pred_raw).detach()
        if res == '2' or res == '0' : ## Retirer la composante SVEN du résidu
            pred_denorm += Y_val_cv
            
        rmse_pow = torch.sqrt(torch.mean((pred_denorm-Y_val_cv)**2))
        rel_rmse_pow = (rmse_pow / torch.sqrt(torch.mean(Y_val_cv**2))) * 100 if torch.mean(Y_val_cv**2) > 1e-5 else 0

        cv_scores[fold] = rel_rmse_pow

    print(f"\n   [2/2] Entraînement complet du Modèle Final (100% Data) sur 1000 époques...")

    ##  Redéclaration de chaque modèle et chaque méthode employée pour éviter
    ##  les résidus d'anciennes sims dans l'entraînement final

    model = PowerMLP(X_train.shape[1], Y_train.shape[1],
                          n_layers = best_params['n_layers'], n_neurons = best_params['n_neurons'],
                          dropout = best_params['dropout_rate'], device = device)
    optimizer = torch.optim.Adam(model.parameters(), lr = best_params['lr'])
    crit = nn.MSELoss()
    best_train_loss = float('inf')

    pbar = tqdm(range(1000), desc=f"   Training Final")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        loss = crit(model(X_train), Y_train)
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()

        if curr_loss < best_train_loss :
            best_train_loss = curr_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0: pbar.set_postfix({"Loss": f"{curr_loss:.6f}"})
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

    ## Début des calculs finaux (pour évaluer le modèle ayant les meilleurs paramètres)
    model.eval()
    with torch.no_grad() :
            preds_raw = model(X_val)
    preds_norm_np = preds_raw.cpu().numpy()

    with open(f"Power_models/Data/DataDir/scalers/scaler_Y_{model_name}_pow.pkl", 'rb') as f : scaler_Y = pickle.load(f)

    if res == '1' or res == '0' :
        Y_denorm = scaler_Y.inverse_transform(Y_val.cpu().numpy())
        pred_denorm = scaler_Y.inverse_transform(preds_norm_np)
        preds_final = pred_denorm + Y_denorm 
    else : 
        preds_final = scaler_Y.inverse_transform(preds_norm_np)
    pow_sven_val = pow_sven_val.cpu().numpy()
    rmse_pow = np.sqrt(np.mean((pow_sven_val - preds_final)**2))
    rel_pow  = (rmse_pow / np.sqrt(np.mean(pow_sven_val**2))) * 100 if np.mean(pow_sven_val) > 1e-5 else 0
    wass_power = wasserstein_distance(preds_final.flatten(), pow_sven_val.flatten())

    results_details = {
            "Model": model_name,
            "Final RMSE" : round(rmse_pow, 4),
            "Final Relative RMSE" : rel_pow,
            "Score 1000 eps" : cv_scores.mean(),
            "Std 1000 eps" : cv_scores.std(),
            "Wass" : wass_power
        }

    os.makedirs("Power_models/performance", exist_ok = True)
    recap_path = 'Power_models/performance/recap_score_glob.csv'
    
    if os.path.exists(recap_path) :
            df_recap = pd.read_csv(recap_path)
            df_recap = df_recap[df_recap["Model"] != "BASELINE_BEM"]
            df_recap = pd.concat([df_recap, pd.DataFrame([results_details])], ignore_index=True)
    else :
            df_recap = pd.DataFrame([results_details])
        
    df_recap.to_csv(recap_path, index = False)
    
    os.makedirs(f"Power_models/models/{entree}", exist_ok=True)
    model_save_path = f"Power_models/models/{entree}/model_{saved_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"    [SAUVEGARDE] Modèle enregistré dans {model_save_path}")

    return

def evaluate_baseline(df_val) :

    P_full = convert_f_to_power(df_val) 

    rmse_pow = np.sqrt(np.mean((P_full['Cp_BEM'].values - P_full['Cp_SVEN'].values)**2))
    rel_rmse = (rmse_pow / np.sqrt(np.mean(P_full['Cp_SVEN'].values**2))) * 100 if np.mean(P_full['Cp_SVEN'].values**2) > 1e-5 else 0
    wass = wasserstein_distance(P_full['Cp_SVEN'].values, P_full['Cp_BEM'].values)
    
    results_baseline = {
         'Model' : 'BASELINE BEM',
         'Final RMSE' : rmse_pow,
         'Final Relative RMSE' : rel_rmse,
         'Score 1000 eps' : 'None',
         'Std 1000 eps' : 'None',
         'Wass' : wass
    }   

    os.makedirs("Power_models/performance", exist_ok = True)
    recap_path = 'Power_models/performance/recap_score_glob.csv'
    
    if os.path.exists(recap_path) :
            df_recap = pd.read_csv(recap_path)
            df_recap = df_recap[df_recap["Model"] != "BASELINE_BEM"]
            df_recap = pd.concat([df_recap, pd.DataFrame([results_baseline])], ignore_index=True)
    else :
            df_recap = pd.DataFrame([results_baseline])

    df_recap.to_csv(recap_path, index=False)
    print(f"    Baseline enregistrée à l'adresse {recap_path}.")

    return

