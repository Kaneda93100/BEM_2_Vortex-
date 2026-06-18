import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

import pathlib as P
import numpy as np
import pandas as pd
import pickle as pkl
import json
import torch
import optuna

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Power_models.src.dataloaders import format_data_power, format_f, get_splits
from Power_models.src.PModels import PowerMLP, ForceEncoder

def optimize_PM(df_train, entree, res, comp, n_trials) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f'{entree}_{res}'
    hp_file_name = model_name

    os.makedirs("HP", exist_ok = True)

    print(f"\n{'='*50}")
    print(f" OPTIMISATION MODÈLE PRÉDICTIF : {model_name}")
    print(f"{'='*50}")    

    ## 1. Préparation des données
    X_set, Y_set = format_data_power(df_train, entree, res, comp, device = device)

    #   -----------------------------------   #
    #   Début fonction objectif pour Optuna   #
    #   -----------------------------------   #
    def objective(trial) :
        lr = trial.suggest_float('lr', 1e-8, 1e-4, log = True)
        n_neurons = trial.suggest_int('n_neurons', 128, 896, step=64)
        dropout_rate = trial.suggest_int('dropout_rate', 0.0, 0.5)
        n_layers = trial.suggest_int('n_layers', 2, 8)

        kf = KFold(n_splits = 3, shuffle = True, random_state = 42)
        cv_scores = []

        model = PowerMLP(X_set.shape[1], Y_set.shape[1],
                              n_layers = n_layers, n_neurons = n_neurons,
                              dropout = dropout_rate, device = device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        crit = torch.nn.MSELoss()
        best_val_loss = float('inf')

        ## Itérer sur les folds 
        for train_idx, val_idx in kf.split(X_set.detach().cpu().numpy()) :
            X_tr, Y_tr = X_set[train_idx], Y_set[train_idx]
            X_val, Y_val = X_set[val_idx], Y_set[val_idx]
            
            ## Entraînement
            for epoch in range(2) :
                model.train()
                optimizer.zero_grad()

                loss = crit(model(X_tr), Y_tr)
                loss.backward(retain_graph = True)
                optimizer.step()
            
                ## Evaluation
                model.eval()
                with torch.no_grad() :
                    val_loss = crit(model(X_val), Y_val).item()
                
                if val_loss < best_val_loss :
                    best_val_loss = val_loss

            cv_scores.append(best_val_loss)

        return sum(cv_scores)/len(cv_scores) ## Score moyenné par le nombre de trial
        #   ---------------------------------   #
        #   Fin fonction objectif pour Optuna   #
        #   ---------------------------------   #

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_model = optuna.create_study(direction = 'minimize')
    study_model.optimize(objective, n_trials = n_trials, show_progress_bar = True)

    ## Sauvegarde des paramètres trouvés
    final_params = {**study_model.best_params}
    
    os.makedirs("Power_models/HP", exist_ok = True)
    file_HP = f"Power_models/HP/{entree}_hp.json"
    if os.path.exists(file_HP) :
            with open(file_HP, "r") as f : all_model_params = json.load(f)
    else :
        all_model_params = {}

    all_model_params[hp_file_name] = final_params
    with open(file_HP, 'w') as f : json.dump(all_model_params,f, indent = 4)
    print(f"   [OK] Modèle {hp_file_name} optimisé.")

    return    


def optimize_AE(df, n_trials, force_opt = False) :

    parms_path = "Power_models/models/fbem_ae.pth"
    path_hp = "Power_models/HP/fbem_ae.json"
    
    if os.path.exists(path_hp) and os.path.exists(parms_path) :
        if not force_opt :
            print(f"[Info]  Un modèle à déjà été trouvé et force_opt == {force_opt}. L'entrainement du compresseur est ignoré.\n")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_model = f'ae'
    os.makedirs("HP_AE", exist_ok = True)
    X = format_f(df)

    #   -----------------------------------   #
    #   Début fonction objectif pour Optuna   #
    #   -----------------------------------   #
    def objective(trial) :
        lr = trial.suggest_float('lr', 1e-8, 1e-4, log = True)
        lat_dim  = trial.suggest_int('lat_dim', 16, 512, step = 16)

        kf = KFold(n_splits = 3, shuffle = True, random_state = 42)
        cv_scores = []
        best_val_loss = float('inf')

        model = ForceEncoder(X.shape[1], latent_dim = lat_dim, device = device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
        crit = torch.nn.MSELoss()
        
        for train_idx, val_idx in kf.split(X.cpu().numpy()) :
            X_train, X_val = X[train_idx], X[val_idx]

            for _ in range(2) :
                model.train()
                optimizer.zero_grad()

                loss = crit(model(X_train), X_train)
                loss.backward()
                optimizer.step()

                
                model.eval()
                with torch.no_grad() :
                    val_loss = crit(model(X_val), X_val).item()
                if val_loss < best_val_loss :
                    best_val_loss = val_loss

            cv_scores.append(best_val_loss)
        return sum(cv_scores)/ len(cv_scores)
        #   ---------------------------------   #
        #   Fin fonction objectif pour Optuna   #
        #   ---------------------------------   #


    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_model = optuna.create_study(direction = 'minimize')
    study_model.optimize(objective, n_trials = n_trials, show_progress_bar = True)

    ## Sauvegarde des paramètres trouvés
    final_params = {**study_model.best_params}
    
    os.makedirs("Power_models/HP", exist_ok = True)
    file_HP = f"Power_models/HP/fbem_ae.json"
    if os.path.exists(file_HP) :
            with open(file_HP, "r") as f : all_model_params = json.load(f)
    else :
        all_model_params = {}

    all_model_params[ae_model] = final_params
    with open(file_HP, 'w') as f : json.dump(all_model_params,f, indent = 4)
    print(f"   [OK] Modèle {ae_model} optimisé.")    

    ## Entraînement et retour des paramètres pour l'encodeur optimal

    opt_compressor = ForceEncoder(X.shape[1], latent_dim = final_params['lat_dim'], device = device)
    optimizer = torch.optim.Adam(opt_compressor.parameters(), lr = final_params['lr'])
    crit = torch.nn.MSELoss()

    opt_compressor.train()
    pbar = tqdm(range(750), desc = " Entraînement du compresseur d'efforts BEM", leave = False)
    for _ in pbar :
        optimizer.zero_grad()
        loss = crit(opt_compressor(X), X)
        loss.backward()
        optimizer.step()
    
    os.makedirs("Power_models/HP/", exist_ok = True)


    if os.path.exists(path_hp) :
        with open(path_hp, "r") as f :
            aes_params = json.load(f)
    else :
        aes_params = {}
    
    with open(path_hp, 'w') as f :
        json.dump(aes_params, f, indent = 4)
    
    parms_path = "Power_models/models/fbem_ae.pth"
    torch.save(opt_compressor.state_dict(), parms_path)

    return
