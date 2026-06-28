import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pathlib as P
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Power_models.src.PModels import ForceEncoder 
from core.physics import compute_cp, compute_density_cp


def get_splits(df, seed = 42, test_size = 0.2, save_dir = None):
    
    yaws_unique = df['yaw'].unique()

    train, val = train_test_split(yaws_unique, test_size = test_size, random_state = seed)

    train_df = df[df['yaw'].isin(train)].copy()
    val_df   = df[df['yaw'].isin(val)].copy()

    if save_dir != None :
        os.makedirs(save_dir, exist_ok = True)
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index = False)
        val_df.to_csv(os.path.join(save_dir, "validation.csv"), index = False)
        print(f" Fichier train.csv et validation.csv enregistré dans le répertoire {save_dir}.")
    
    return train_df, val_df
def convert_f_to_power(df) :

    ## Récupérer les colonnes contenant les forces (ajouter une partie permettant de traiter les vitesses et les angles d'attaques)
    fns = [key for key in df.keys() if key in ['Fn_BEM', 'Fn_SVEN']]
    fts = [key for key in df.keys() if key in ['Ft_BEM', 'Ft_SVEN']]
    
    P_bem   = compute_cp(df, fns[0], fts[0])
    P_sven  = compute_cp(df, fns[1], fts[1])
    ds_full = pd.merge(P_bem, P_sven, on = ['yaw', 'TSR'])

    return ds_full

def format_data_power(df, entree, res, comp, scaler_exist = True, device = 'cpu') :


    ## 1. Choix de l'approche residuelle
    if entree == 'DP' :

        P_full = convert_f_to_power(df)
        P = P_full[['yaw', 'TSR', 'Cp_BEM', 'Cp_SVEN']]
        
        if res == '2' :         ## Features == [yaw, tsr, BEM] | target = [SVEN]
            X = P[['yaw', 'TSR', 'Cp_BEM']].values
            Y = P[['Cp_SVEN']].values
        elif res == '1' :       ## Features == [yaw, tsr, BEM] | target = [SVEN-BEM]
            X = P[['yaw', 'TSR', 'Cp_BEM']].values
            Y = P['Cp_BEM'].values - P['Cp_SVEN'].values
        elif res == '0' :       ## Features == [yaw, tsr] | target = [SVEN]
            X = P[['yaw', 'TSR']].values
            Y = P['Cp_SVEN'].values
        elif res == '-1' :      ## Features == [yaw, tsr] | target = [SVEN - BEM]
            X = P[['yaw', 'TSR']].values
            Y = P['Cp_BEM'].values - P['Cp_SVEN'].values
        Y = Y.reshape(-1,1)


    elif entree == 'GVP' :
        df_calc = df.copy()
        grand_group = df_calc.groupby(['yaw', 'TSR'])
        X,Y = [],[]
        
        for (y_val, tsr_val), group in grand_group :
            dQ_bem = compute_density_cp(group['Fn_BEM'].values, group['Ft_BEM'].values)
            dQ_sven = compute_density_cp(group['Fn_SVEN'].values, group['Ft_SVEN'].values)
            forces_bem = group[['Fn_BEM', 'Ft_BEM']].values.flatten()

            if res == '2' :      ## Features == [yaw, tsr, BEM] | target = [SVEN]
                X_val = np.concatenate(([y_val, tsr_val], forces_bem))
                Y_val = dQ_sven
            elif res == '1' :    ## 
                X_val =  np.concatenate(([y_val, tsr_val], forces_bem))
                Y_val = dQ_bem - dQ_sven
            elif res == '0' :
                X_val = [y_val, tsr_val]
                Y_val = dQ_sven
            elif res == '-1' :
                X_val = [y_val, tsr_val]
                Y_val = dQ_bem - dQ_sven

            X.append(X_val)
            Y.append(Y_val)
        
        X, Y = np.array(X), np.array(Y)
    
    elif entree == 'GMP' :
        df_calc = df.copy()
        grand_group = df_calc.groupby(['yaw', 'TSR'])
        X,Y = [],[]

        r = np.sort(df_calc['r'].unique())
        theta = np.sort(df_calc['theta'].unique())

        for (y_val, tsr_val), group in grand_group :
            dQ_bem = compute_density_cp(group['Fn_BEM'].values, group['Ft_BEM'].values)
            dQ_sven = compute_density_cp(group['Fn_SVEN'].values, group['Ft_SVEN'].values)
            bem_img = group[['Fn_BEM', 'Ft_BEM']].values.reshape(len(r), len(theta),2)

            yaw_channel = np.full((len(r), len(theta)),y_val)
            tsr_channel = np.full((len(r), len(theta)), tsr_val)

            if res == '2' :
                Y_val = dQ_sven
                X_val = np.stack([bem_img[:,:,0], bem_img[:,:,1], yaw_channel, tsr_channel])
            elif res == '1' :
                Y_val = dQ_bem - dQ_sven
                X_val = np.stack([bem_img, yaw_channel, tsr_channel])
            else :
                raise Exception(f"Approche GM sans BEM en entrée non supportée.\n")
            X.append(X_val)
            Y.append(Y_val)
        X_np = np.array(X)
        Y_np = np.array(Y)


    ## 2. Normalisation
    model_name = f"{entree}_{res}_{comp}"
    os.makedirs("Power_models/scalers", exist_ok = True)
    path_x, path_y = f"Power_models/scalers/scaler_X_{model_name}.pkl", f"Power_models/scalers/scaler_Y_{model_name}.pkl"  

    if f"scaler_X_{model_name}.pkl" not in os.listdir("Power_models/scalers") and f"scaler_Y_{model_name}.pkl" not in os.listdir("Power_models/scalers") :
        scaler_exist = False

    if entree == 'GVP' or entree == 'DP': 
        if scaler_exist :
            print("\n Des scalers ont été trouvé.\n")
            with open(path_x, 'rb') as f :
                scaler_X = pkl.load(f)
            with open(path_y, 'rb') as f :
                scaler_Y = pkl.load(f)
            X_scaled = scaler_X.transform(X)
            Y_scaled = scaler_Y.transform(Y)
        else : 
            scaler_X, scaler_Y = StandardScaler(), StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            Y_scaled = scaler_Y.fit_transform(Y)

            with open(path_x, 'wb') as f :
                pkl.dump(scaler_X, f)
            with open(path_y, 'wb') as f : 
                pkl.dump(scaler_Y, f)

    elif entree == 'GMP':
        ## Applatir l'image sur tous les canaux pour normaliser
        X_original_shape = X_np.shape
        X_np = X_np.reshape((X_np.shape[0], X_np.shape[1]*X_np.shape[2]*X_np.shape[3]))

        if scaler_exist :
            print("\n Des scalers ont été trouvé.\n")
            with open(path_x, 'rb') as f :
                scaler_X = pkl.load(f)
            with open(path_y, 'rb') as f :
                scaler_Y = pkl.load(f)
            
            X_scaled = scaler_X.fit_transform(X_np)
            Y_scaled = scaler_Y.fit_transform(Y_np)
        else :
            print("Aucun scaler n'a été trouvé. Ils vont être calculés.\n")
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()

            X_scaled = scaler_X.fit_transform(X_np)
            Y_scaled = scaler_Y.fit_transform(Y_np)

            with open(path_x, 'wb') as f :
                pkl.dump(scaler_X, f)
            with open(path_y, 'wb') as f :
                pkl.dump(scaler_Y,f)

    X_scaled = X_scaled.reshape(X_original_shape)
    X_tensor = torch.tensor(X_scaled, dtype = torch.float32, device = device)
    Y_tensor = torch.tensor(Y_scaled, dtype = torch.float32, device = device)
    
    ## 3. Gestion de la compression des efforts BEM (seulement pour GVP pour l'instant)
    if entree == 'GVP' :
        if res == '2' or res == '1' : ## Pas de vecteurs de force dans les autres cas
            if comp == True :
                path_to_comp_params = "Power_models/models/fbem_ae.pth"
                path_to_comp_hp = "Power_models/HP/fbem_ae.json"
                if os.path.exists(path_to_comp_params) and os.path.exists(path_to_comp_hp) :
                    with open(path_to_comp_hp, 'r') as f : 
                        ae_hp = json.load(f)
                    compressor = ForceEncoder(X.shape[1], latent_dim = ae_hp['ae']['lat_dim'], device = device)
                    compressor.load_state_dict(torch.load(path_to_comp_params, weights_only = False))
                    print(f"[Info]   Compresseur chargé. Dimension latente : {ae_hp['ae']['lat_dim']}\n")
                else :
                    raise Exception(f"\nLe fichier contenant les hyperparamètres ou les paramètres du compresseur n'ont pas été trouvé. Lancez la procédure 'optimise_AE' avant.\n")
                compressor.eval()
                X_tensor = compressor.encode(X_tensor).detach()
    
    return X_tensor, Y_tensor

def format_f(df, scaler_exist = False) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    df_cop = df[['yaw', 'TSR', 'Fn_BEM', 'Ft_BEM']].copy()
    grand_group = df_cop.groupby(['yaw', 'TSR'])
    features = []

    for (yaw, tsr), group in grand_group :
        f_flat = group[['Fn_BEM', 'Ft_BEM']].values.flatten()
        f_flat = np.concatenate(([yaw, tsr], f_flat))
        features.append(f_flat)
    X_np = np.array(features)

    model_name = f"fbem_scalers"
    os.makedirs("Power_models/scalers_FBEM", exist_ok = True)
    path_fbem = P.Path(f"Power_models/scalers_FBEM/{model_name}.pkl")
    
    if not os.path.exists(path_fbem) :
        scaler_exist = False
        
    if scaler_exist:
        if not os.exists(path_fbem) :
            raise ValueError(f"Les scalers n'ont pas été trouvés. Régler l'option scaler_exist = False pour les calculer et les enregistrer.\n")
        else :
            print("\nDes scalers ont été trouvé !\n")

        with open(path_fbem, 'rb') as f :
            scaler = pkl.load(f)
        X_scaled = scaler.fit_transform(X_np)
    else :
        print(f"\nscaler_exist == {scaler_exist}, des scalers vont être calculés et enregistré à l'adresse {path_fbem}.\n")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)

        with open(path_fbem, 'wb') as f :
            pkl.dump(scaler, f)

    return torch.tensor(X_scaled, dtype = torch.float32, device = device)


