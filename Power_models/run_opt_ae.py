import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

from pyplot import matplotlib as plt
from Power_models.src.evaluator import evaluator_power, evaluate_baseline
from Power_models.src.optimize import  optimize_AE
from Power_models.src.PModels import ForceEncoder
from Power_models.src.dataloaders import get_splits
from Power_models.src.PModels import ForceEncoder

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch
import numpy as np
import json
import pathlib as P
import pandas as pd
from time import perf_counter

path_data = P.Path('data/raw/fichier_forces.csv')

## Paramètres de l'optimisation

n_trials = 500
eps_obj = 1250
eps_train = 1250

def run_opt_ae() : 
    print("[Début]  Lancement de la procédure d'optimisation et d'analyse des performances de l'encodeur d'effort BEM.\n")
    
    ## Chargement des données
    df = pd.read_csv(path_data)

    start_opt = perf_counter()
    optimize_AE(df, n_trials = n_trials, eps_obj = eps_obj, eps_train = eps_train, force_opt = True)
    stop_opt = perf_counter()

    total_exe = round((stop_opt-start_opt)/60, 3)

    print(f"\n\n[Fin]    Optimisation terminée en {total_exe} minutes.\n")

    return

def load_model(path_hp, path_par) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Charger les hyperparamètres
    try :
        with open(path_hp, 'r') as f :
            hp_dict = json.load(f)
    except :
        Exception(f"Impossible d'ouvrir le fichier .json à l'adresse {path_hp}\n")
    
    ## Déclarer le modèle
    lat_dim = hp_dict['ae']['lat_dim']
    model = ForceEncoder(latent_dim = lat_dim, device = device)

    ## Charger les poids et les biais
    model.load_state_dict(torch.load(path_par))

    print(f"Modèle chargé avec succès sur {device}.\n", model)

    return model

def esp_std(mdl, df) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if mdl.device != device : raise Exception(f"{mdl.device} insdisponible.\n"); return

    ## Charger le modèle

    ## Récupérer les distributions de forces sous forme de vecteur
    df_cop = df[['yaw', 'TSR', 'Fn_BEM', 'Ft_BEM']].copy()
    grand_group = df_cop.groupby(['yaw', 'TSR'])
    features = []

    for (yaw, tsr), group in grand_group :
        f_flat = group[['Fn_BEM', 'Ft_BEM']].values.flatten()
        features.append(f_flat)
    X = torch.tensor(features, dtype = torch.float32, device = device)
    
    big_batch = DataLoader(TensorDataset(X), batch_size = 1000, pin_memory = False, shuffle = True)

    ## Évaluation du modèle
    
    Av_err = []
    Va_err = []
    Rel_av_err = []

    mdl.eval()
    for batch in big_batch : 
        pred = mdl(batch)

        av =  torch.sum(torch.abs(pred-batch))/len(batch)
        va =  torch.sum(torch.abs(pred-batch)**2)/len(batch) - av**2
        rel_av = torch.sum(torch.abs(pred-batch))/torch.sum(torch.abs(batch))


        Av_err.append(av.detach().cpu().item())
        Va_err.append(va.detach().cpu().item())
        Rel_av_err.append(rel_av.detach().cpu().item())

    Average_error = Av_err.mean()
    Variance_error = Va_err.mean()
    Relative_error = Rel_av_err.mean()

    return Average_error, Variance_error, Relative_error


def plot(az, tsr, df, mdl) :
    
    df_tsr = df[df['TSR'] == tsr]
    blade = df_tsr[df_tsr['theta'] == az]
    blade.sort(on = 'r')
    radius = df['r'].sort().unique()
    
    ff = torch.tensor(blade[['Fn_BEM', 'Ft_BEM']].values.flatten(), dtype = torch.float32)
    Fn = blade['Fn_BEM']; Ft = blade['Ft_BEM']

    mdl.to('cpu').eval()
    ff_ae = mdl(ff)
    ff_ae = ff_ae.reshape((2,1,2592))
    Fn_ae = ff_ae[0,0,:], Ft_ae = ff_ae[1,0,:]

    fig = plt.figure(figsize = (20,20))

    plt.subplot(1,2,1)
    plt.plot(radius, Fn, color = 'blue', label = 'BEM')
    plt.plot(radius, Fn_ae, color = 'red', label = 'Ae')
    plt.title(f"Fn, sur l'azimuth {az}")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("N/m")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(radius, Ft, color = 'blue', label = 'BEM')
    plt.plot(radius, Ft_ae, color = 'red', label = 'Ae')
    plt.title(f"Ft, sur l'azimuth {az}")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("N/m")
    plt.grid()

    plt.show()

    return

if __name__ == '__main__' :
    ## 1 trials à 1000 eps (obj et train) --> 0.403 minutes, 
    ## pour 500 trials --> 3,35 heures

    ## 1 trials à 1250 eps (obj et train) --> 0.532 minutes
    ## pour 500 trials --> 4,43 heures

    ## 1 trials à 1500 eps (obj et train) --> 0.566 minutes,
    ## pour 500 trials --> 4,71 heures

    ## 1 trials à 2000 eps (obj et train) --> 0.8 minutes
    ## pour 500 trials --> 6,67 heures

    run_opt_ae()
    df = pd.read_csv(path_data)
    mdl = load_model(path_hp ="Power_models/HP/fbem_ae.json" ,path_para = 'Power_models/models/fbem_ae.pth')

    print("\n\n",esp_std(mdl, df),"\n\n")

    plot(az = 35.0, tsr = df['TSR'][0], mdl = mdl)