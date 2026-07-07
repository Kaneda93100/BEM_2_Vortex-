import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

import json
import pickle as pkl
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
from Power_models.src.PModels import PowerMLP, PowerCNN
from core.models import TorchScaler
from core.physics import compute_cp, compute_cp_diff, convert_v_to_f
from Power_models.src.dataloaders import convert_f_to_power
from training.src.data_loader import format_data

def evaluator_power(df_train, df_val, entree, res, comp, eps_cv = 1000, eps_train = 1000) :
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = f'{entree}_{res}_{comp}'
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

    ## Récupérer le scaler pour les labels 
    get_scale = f'scaler_Y_{model_name}.pkl' 
    if get_scale != None : 
        try :
            with open(f"Power_models/scalers/{get_scale}", 'rb') as f :
                scaler_label = pkl.load(f)
        except FileNotFoundError :
            print(f"Le scaler {get_scale} n'a pas été trouvé. Il va être calculé.\n")
            df = pd.read_csv("data/raw/fichier_forces.csv")
            _,_ = format_data_power(df_train, entree = entree, res = res, is_train = False)
            with open(f"Power_models/scalers/{get_scale}", 'rb') as f :
                scaler_label = pkl.load(f)

    
    if entree == 'DP' or entree == 'GVP' :
        model = PowerMLP(X_train.shape[1], Y_train.shape[1],
                            n_layers = best_params['n_layers'], n_neurons = best_params['n_neurons'],
                            dropout = best_params['dropout_rate'], device = device)
    elif entree == 'GMP' :
        model = PowerCNN(X_train.shape[1], n_layers = best_params['n_layers'], base_filters = best_params['base_filters'],
                         dropout_rate = best_params['dropout_rate'], size_output = Y_train.shape[1], device = device) 
    
    """
    ## Récupérer les puissances de références pour la Cross-Validation
    _, pow_sven_val = format_data_power(df_val, entree, res = '0', comp = False, scaler_exist = False, device = device) 
    with open(f'Power_models/scalers/scaler_Y_{entree}_0_False.pkl', 'rb') as f :
        scaler_Y = pkl.load(f)
    scaler_TPow_torch = TorchScaler(scaler_Y, device = device)

    ## scaler_field
    get_scale = f'scaler_Y_GV_0_f.pkl' if entree == 'GVP' else None
    if get_scale != None : 
            try :
                with open(f"training/scalers/{get_scale}", 'rb') as f :
                    scaler_field = pkl.load(f)
            except FileNotFoundError :
                print(f"Le scaler {get_scale} n'a pas été trouvé. Il va être calculé.\n")
                df = pd.read_csv("data/raw/fichier_forces.csv")
                _,_ = format_data(df_train, entree = 'GV', res = '0', inter = 'f', is_train = True)
                with open(f"training/scalers/{get_scale}", 'rb') as f :
                    scaler_field = pkl.load(f)
            finally :
                scaler_field_torch = TorchScaler(scaler_field, device = device) 

    if entree == 'GVP' :
        ## scaler_scalar
        with open(f"Power_models/scalers/scaler_Y_{model_name}.pkl" ,'rb') as f :
                scaler_scalar = pkl.load(f)
                scaler_scalar_torch = TorchScaler(scaler_scalar, device = device)
        
        ## scaler_field
        get_scale = f'scaler_Y_GV_0_f.pkl' if entree == 'GVP' else None
        if get_scale != None : 
            try :
                with open(f"training/scalers/{get_scale}", 'rb') as f :
                    scaler_field = pkl.load(f)
                    scaler_field_torch = TorchScaler(scaler_field, device = device) 
            except FileNotFoundError :
                print(f"Le scaler {get_scale} n'a pas été trouvé. Il va être calculé.\n")
                _,_ = format_data(df_train, entree = 'GV', res = '0', inter = 'f', is_train = True)
                with open(f"training/scalers/{get_scale}", 'rb') as f :
                    scaler_field = pkl.load(f)
            finally :
                scaler_field_torch = TorchScaler(scaler_field, device = device) 

    """    

         
    print("\n   [1/2] Lancement de la Cross-Validation (3 Folds x 1000 époques)...")
        
    n_splits = 3
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=42)

    cv_scores = np.zeros(n_splits)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train.cpu().numpy())) :
        X_tr_cv, Y_tr_cv    = X_train[train_idx], Y_train[train_idx]
        X_val_cv, Y_val_cv  = X_train[val_idx], Y_train[val_idx]

        optimizer = torch.optim.Adam(model.parameters(), lr = best_params['lr'])
        crit = nn.MSELoss()
        pbar = tqdm(range(eps_cv), desc=f"   -> Fold {fold+1}/{n_splits}", leave=False)

        for epoch in pbar :
            model.train()
            optimizer.zero_grad()
            
            loss = crit(model(X_tr_cv), Y_tr_cv)
            loss.backward()
            optimizer.step()

        model.eval()
        pred_raw = model(X_val_cv)
        if res == '1' or res == '-1' : ## Retirer la composante BEM du résidu
            pred_raw += Y_val_cv

        pred_denorm = scaler_label.fit_transform(pred_raw.detach().to('cpu').numpy())
        Y_val_cv_denorm = scaler_label.fit_transform(Y_val_cv.detach().to('cpu').numpy())

        rmse_pow = np.sqrt(np.mean((pred_denorm-Y_val_cv_denorm)**2))
        rel_rmse_pow = (rmse_pow / np.sqrt(np.mean(Y_val_cv_denorm**2))) * 100 if np.mean(Y_val_cv_denorm**2) > 1e-5 else 0

        cv_scores[fold] = rel_rmse_pow

    print(f"\n[2/2]     Entraînement complet du Modèle Final (100% Data) sur 1000 époques...")

    ##  Redéclaration de chaque modèle et chaque méthode employée pour éviter
    ##  les résidus d'anciennes sims dans l'entraînement final

    if entree == 'DP' or entree == 'GVP' :
        model = PowerMLP(X_train.shape[1], Y_train.shape[1],
                            n_layers = best_params['n_layers'], n_neurons = best_params['n_neurons'],
                            dropout = best_params['dropout_rate'], device = device)
    elif entree == 'GMP' :
        model = PowerCNN(X_train.shape[1], n_layers = best_params['n_layers'], base_filters = best_params['base_filters'],
                         dropout_rate = best_params['dropout_rate'], size_output = Y_train.shape[1], device = device) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr = best_params['lr'])
    crit = nn.MSELoss()
    best_train_loss = float('inf')

    pbar = tqdm(range(eps_train), desc=f"   Training Final")
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



    ## Récupérer les données de références pour calculer l'erreur relative et pour ajuster le rédidu si besoins est
    if res == '1' or res == '-1' :
        if entree == 'DP' : 
            Y_sven = compute_cp(df_val,'Fn_SVEN', 'Ft_SVEN')['Cp_SVEN'].values
        elif entree == 'GVP' or entree == 'GMP' :
            val_group = df_val.groupby(['yaw', 'TSR'])

            dQ = []
            for (_,_), group in val_group :
                dQ_az = []
                f = group[['r','theta','Fn_SVEN', 'Ft_SVEN']]
                theta_group = f.groupby(['theta'])
                for (_), blade in theta_group :
                    blade = blade.sort_values(by = 'r')          
                    blade = blade[['Fn_SVEN', 'Ft_SVEN']]
                    torch_dQ = compute_cp_diff(
                        torch.tensor(blade['Fn_SVEN'].values, dtype = torch.float32, device = 'cpu'),
                        torch.tensor(blade['Ft_SVEN'].values, dtype = torch.float32, device = 'cpu')
                    )
                    dQ_az.append(torch_dQ)
                dQ_az.append(dQ)
                dQ = []                   
            Y_sven = torch.tensor(dQ, dtype = torch.float32, device = 'cpu').numpy().flatten()
        
        with open(f"Power_models/scalers/scaler_Y_{model_name}.pkl", 'rb') as f : 
            scaler_Y = pkl.load(f)
        pred_denorm = scaler_Y.inverse_transform(preds_norm_np).flatten()
        cp_bem      = compute_cp(df_val,'Fn_BEM', 'Ft_BEM')['Cp_BEM'].values        
        preds_final = (pred_denorm + cp_bem)
    else : ## res == 2 ou 0
        with open(f"Power_models/scalers/scaler_Y_{model_name}.pkl", 'rb') as f : 
            scaler_Y = pkl.load(f)
        Y_sven = scaler_Y.inverse_transform(Y_val.cpu().numpy())
        preds_final = scaler_Y.inverse_transform(preds_norm_np)
    
    rmse_pow = np.linalg.norm(Y_sven - preds_final)
    rel_pow  = (rmse_pow / np.linalg.norm(Y_sven)) * 100 if np.linalg.norm(Y_sven) > 1e-10 else 0
    wass_power = wasserstein_distance(preds_final.flatten(), Y_sven.flatten())

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
            
            if results_details['Model'] in df_recap['Model'].values :
                i = df_recap[df_recap['Model'] == results_details['Model']].index
                df_recap.loc[i] = list(results_details.values())
            else :
                df_recap = pd.concat([df_recap, pd.DataFrame([results_details])], ignore_index=True)
    else :
            df_recap = pd.DataFrame([results_details])
    df_recap.to_csv(recap_path, index = False)

    os.makedirs(f"Power_models/models/{entree}", exist_ok=True)
    model_save_path = f"Power_models/models/{entree}/model_{saved_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"[SAUVEGARDE]     Modèle enregistré dans {model_save_path}")

    return

def evaluate_baseline(df_val) :

    P_full = convert_f_to_power(df_val) 

    rmse_pow = np.sqrt(np.mean((P_full['Cp_BEM'].values - P_full['Cp_SVEN'].values)**2))
    rel_rmse = (rmse_pow / np.sqrt(np.mean(P_full['Cp_SVEN'].values**2))) * 100 if np.mean(P_full['Cp_SVEN'].values**2) > 1e-9 else 0
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
            
            if results_baseline['Model'] in df_recap['Model'].values :
                i = df_recap[df_recap['Model'] == results_baseline['Model']].index[0]
                df_recap.loc[i] = results_baseline
            else :
                df_recap = pd.concat([df_recap, pd.DataFrame([results_baseline])], ignore_index=True)
    else :
            df_recap = pd.DataFrame([results_baseline])

    df_recap.to_csv(recap_path, index=False)
    print(f"    Baseline enregistrée à l'adresse {recap_path}.")

    return

