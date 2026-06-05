import sys
import os
import time

dir_path = os.path.dirname(sys.path[0])
sys.path.append(dir_path)

import pathlib as P
import numpy as np
import pandas as pd
import json
import pickle

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from scipy.stats import wasserstein_distance
import optuna

from src.data_loader import get_splits
from src.models import TurbineMLP 
from src.evaluate import reconstruct_predictions
from src.physics import convert_v_to_f

datas_dir = P.Path('DataSet/Final_DS')

def format_data(df, entree, res, inter) :

    res_str = str(res)

    # 1. Sélection des variables intermédiaires
    if inter == 'f':
        cols_sven, cols_bem = ['Fn_SVEN', 'Ft_SVEN'], ['Fn_BEM', 'Ft_BEM']
    elif inter == 'v':
        cols_sven, cols_bem = ['V_eff_SVEN', 'alpha_SVEN'], ['V_eff_BEM', 'alpha_BEM']
    else:
        raise ValueError(f"Intermédiaire '{inter}' non reconnu.")

    # 2. Sélection des données selon la stratégie
    if entree == 'L':
        indexs = df[cols_sven].index
        if res_str == '1': # Stratégie résiduelle
            if ponderate == True : 
                Ypond = []
                for i in range(len(cols_sven))  :
                    Ypond.append((df[cols_sven[i]].values - df[cols_bem[i]].values)*df['r'][indexs].values)
                Y_np = np.array(Ypond).T
                X_np = df[['r', 'cos_theta', 'sin_theta', 'yaw'] + cols_bem].values
            else : 
                Y_np = df[cols_sven].values - df[cols_bem].values
                X_np = df[['r', 'cos_theta', 'sin_theta', 'yaw'] + cols_bem].values
        else: # Stratégie directe
            if ponderate == True : 
                Ypond = []
                for i in range(len(cols_sven))  :
                    Ypond.append((df[cols_sven[i]].values)*df['r'][indexs].values)
                Y_np = np.array(Ypond).T
                X_np = df[['r', 'cos_theta', 'sin_theta', 'yaw'] + cols_bem].values
            else : 
                Y_np = df[cols_sven].values
                X_np = df[['r', 'cos_theta', 'sin_theta', 'yaw']].values
    elif entree == 'G': 
        grouped = df.groupby('yaw')
        X_list, Y_list = [], []
        for y_val, group in grouped:
            # Tri strict pour que le vecteur soit toujours dans le même ordre (theta puis r)
            group = group.sort_values(['theta', 'r'])
            index_group = group.index

            if ponderate == True :
                Ypond_sven = []
                Ypond_bem = []
                for i in range(len(cols_sven)) :
                    Ypond_sven.append(group[cols_sven[i]].values*df['r'][index_group].values)
                    Ypond_bem.append(group[cols_bem[i]].values*df['r'][index_group].values) 
                y_sven = (np.array(Ypond_sven)).flatten()
                y_bem = (np.array(Ypond_bem)).flatten()
            else : 
                y_sven = group[cols_sven].values.flatten()
                y_bem = group[cols_bem].values.flatten() 
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.concatenate(([y_val], y_bem)) 
            else:
                Y_val = y_sven
                X_val = np.array([y_val])    
    return 

def optimization_process(X,Y, model_name, n_trials = 1) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    X = X.to(device)
    Y = Y.to(device)

    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 2, 8)
        n_neurons = trial.suggest_int('n_neurons', 64, 512)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log = True)

        k = KFold(n_splits = 3, shuffle = True, random_state = 42)
        cv_scores = []
        criterion = nn.MSELoss()

        if 'G' in model_name :
            model_fam = 'G'
        elif 'L' in model_name :
            model_fam = 'L'
        else :
            raise ValueError(f"Le nom du modèle {model_name} est invalide.\n")
        
        for train_idx, val_idx in k.split(range(len(X))) :
            X_tr, Y_tr = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
            
            if model_fam == 'G':
                b_size = len(X_tr) 
            else :  # 'L'
                b_size = 1024

            train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=b_size, shuffle=True)

            model = TurbineMLP(X.shape[1], Y.shape[1], n_layers, n_neurons, dropout_rate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)

            ## Entrainement
            model.train()
            for epoch in range(1) :
                for batch_x, batch_y in train_loader :
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
            
            ## Validation
            model.eval()
            with torch.no_grad() :
                val_loss = criterion(model(X_val), Y_val)
            cv_scores.append(val_loss)
    
        return sum(cv_scores) / len(cv_scores)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective, n_trials = n_trials)
    os.makedirs('hp_FT', exist_ok = True)
    with open(f"hp_FT/{model_name}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"   Modèle {model_name} optimisé sur {device}. Best MSE: {study.best_value:.4f}")
    return

def train_val_save(df_train, df_val, residuelle, file_name, inter, model_name, epochs = 2) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'G' in model_name : 
        fam = 'G'
    elif 'L' in model_name :
        fam = 'L'
    else : 
        raise ValueError(f"\nLe nom {model_name} ne contient pas d'indicatif d'appartenance à une famille de stratégie (G ou L). Arrêt.\n" )
    
    if 'nc' in model_name :
        comp = 'nc'
    elif 'c' in model_name :
        comp = 'c'
    else :
        raise ValueError(f"\nLe nom {model_name} ne contient pas d'indicatif de compression. Arrêt.\n")

    # Charger les hyperparamètres
    hp_path = f"hp_FT/{model_name}.json"
    try : 
        with open(hp_path, "r") as f: hparams = json.load(f)
    except FileNotFoundError :
        print(f"Le dossier {hp_path} n'a pas été trouvé. Arrêt.\n")
        exit()

    if fam == 'G' :
        features_keys = ['yaw', 'TSR']
        targ_keys = []
        for key in df_train.keys() :
            if 'BEM' in key : features_keys.append(key)
            if 'SVEN' in key : targ_keys.append(key)
    elif fam == 'L' :
        features_keys = ['yaw', 'TSR', 'r', 'theta']
        targ_keys = []
        for key in df_train.keys() :
            if 'BEM' in key : features_keys.append(key)
            if 'SVEN' in key : targ_keys.append(key)

    X_train = torch.tensor(df_train[features_keys].values, dtype = torch.float32, device = device)
    Y_train = torch.tensor(df_train[targ_keys].values, dtype = torch.float32, device = device)

    X_val = torch.tensor(df_val[features_keys].values, dtype = torch.float32, device = device)
    Y_val = torch.tensor(df_val[targ_keys].values, dtype = torch.float32, device = device)

    model = TurbineMLP(X_train.shape[1], Y_train.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()

    if fam == 'G' :
        b_size = len(X_train)
    elif fam == 'L' :
        b_size = 1024
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size = b_size, shuffle = True)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad(): 
        preds_raw = model(X_val).cpu().numpy()
        
    test_loss_mse = np.mean((preds_raw - Y_val.to('cpu').numpy())**2)    
    
    try : 
        with open(f"FatTraining/scalers/{file_name}_S.pkl", 'rb') as f: scaler_Y = pickle.load(f)
    except FileNotFoundError :
        print(f"\nLe dossier scalers/ ou {file_name}_S.pkl n'a pas été trouvé. Arrêt.\n")
        exit()
    preds_denorm = scaler_Y.inverse_transform(preds_raw)
    
    df_res = reconstruct_predictions(df_val, preds_denorm, fam, residuelle, inter, comp = comp) 

    if inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)
    
    # 7. Métriques Finales
    if comp == 'nc' :
        Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
        Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values

        rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
        rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
        rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
        rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
        wass_fn = wasserstein_distance(Fn_s, Fn_p)
        wass_ft = wasserstein_distance(Ft_s, Ft_p)

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
        
        if inter == 'v':
            results_detail["RMSE_Veff_Abs"] = np.sqrt(np.mean((df_res['V_eff_pred'].values - df_res['V_eff_SVEN'].values)**2))
            results_detail["RMSE_Alpha_Abs"] = np.sqrt(np.mean((df_res['alpha_pred'].values - df_res['alpha_SVEN'].values)**2))
    
        recap_data = {
            "Modele": model_name,
            "Epochs_Conv": epochs,
            "Score_Global_%": rel_fn + rel_ft,
            "RMSE_Fn_Rel_%": rel_fn,
            "RMSE_Ft_Rel_%": rel_ft,
            "Wasserstein_Fn": wass_fn,
            "Wasserstein_Ft": wass_ft
        }


    else :
        sven = df_res['SVEN'].values
        pred = df_res['pred'].values

        rmse = np.sqrt(np.mean((pred - sven)**2))
        rel_rmse = rmse/np.sqrt(np.mean(sven)) * 100 if np.sqrt(np.mean(sven)) > 1e-5 else 0
        wass = wasserstein_distance(pred, sven) ## Choix discutable (on compare la distribution des fn/ft compressée à celle prédite)

        recap_data = {
            'Modele' : model_name,
            'Epoch' : epochs,
            'RMSE' : rmse,
            'RELATIVE RMSE [%]' : rel_rmse,
            'WASSERSTEIN' : wass
        }         

        results_detail = recap_data

    #model_name = model_name + '_' + residuelle + '_' + file_name 
    os.makedirs("FatTraining/performance", exist_ok=True)
    pd.DataFrame([results_detail]).to_csv(f"FatTraining/performance/results_{model_name}.csv", index=False)

    os.makedirs("FatTraining/performance", exist_ok = True)
    recap_path = "FatTraining/performance/recap_scores_globaux.csv"
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != model]
        df_recap = pd.concat([df_recap, pd.DataFrame([recap_data])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([recap_data])
    
    if comp == 'nc' :
        df_recap = df_recap.sort_values(by="Score_Global_%", ascending=True).reset_index(drop=True)
        df_recap.to_csv(recap_path, index=False)
    else :
        df_recap.to_csv(recap_path, index = False)

    print(f" Terminé ! ")
    return 

def FatTraining(datas_dir = datas_dir) :
    
    for csv in os.listdir(datas_dir) :
        csv = '/home/arthur/Documents/GitHub/BEM_2_Vortex-/DataSet/Final_DS/BEM_Dummy_f_p_nc.csv'
        file_name = P.Path(csv).stem
        if '_v_' in P.Path(csv).stem :
            inter = 'v'
        elif '_f_' in P.Path(csv).stem :
            inter = 'f'
    
        for model in ['G','L'] :
            for res in ['0', '1'] :
                name_model = model + '_' + P.Path(csv).stem
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                df = pd.read_csv(datas_dir/csv)
                df_train, df_val = get_splits(df, entree = model)

                if model == 'G' :
                    features_keys = ['yaw', 'TSR']
                    targ_keys = []
                    for key in df_train.keys() :
                        if 'BEM' in key : features_keys.append(key)
                        if 'SVEN' in key : targ_keys.append(key)
                elif model == 'L' :
                    features_keys = ['yaw', 'TSR', 'r', 'theta']
                    targ_keys = []
                    for key in df_train.keys() :
                        if 'BEM' in key : features_keys.append(key)
                        if 'SVEN' in key : targ_keys.append(key)

                X = torch.tensor(df_train[features_keys].values, dtype = torch.float32, device = device)
                Y = torch.tensor(df_train[targ_keys].values, dtype = torch.float32, device = device)

                print("-"*85)
                print("-"*85)
                print("-"*85)
                print(f"----- Début de l'optimisation des hyperparamètres, classe de modèle {model} -----")

                start_opt = time.perf_counter()
                optimization_process(X = X, Y = Y, model_name = name_model)
                stop_opt = time.perf_counter()

                opt_perf = stop_opt-start_opt

                print(f"----- Optimisation terminée ! Réalisée en {opt_perf} sec ({(opt_perf)/60} minutes, {(opt_perf)/3600} heures) -----")
                print("-"*85)

                print("\n")

                print("-"*85)
                print(" ----- Début de l'évaluation et de l'entrainement allongé -----")
                
                start_train = time.perf_counter()
                train_val_save(df_train, df_val, file_name = file_name, residuelle = res, inter = inter, model_name = name_model)
                stop_train = time.perf_counter()

                train_perf = stop_train - start_train

                print(f"----- Entrainement terminée ! Réalisée en {train_perf} sec ({(train_perf)/60} minutes, {(train_perf)/3600} heures) -----")
                print("-"*85)
                print(f"Entrainement et optimisation réalisée en {train_perf + opt_perf} sec ({(train_perf + opt_perf)/60} minutes)\n")
                print("-"*85)
                print("-"*85)
                print("-"*85)

                print("\n")
    
    return

if __name__ == "__main__" :
    FatTraining()