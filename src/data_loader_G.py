import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

"""
DataLoader pour la stratégie "G", pour "Globale". Les grilles d'azimuths et de rayons sont fixés, et les données qui serviront à l'entraînement
seront : 
    - Le yaw
    - Le TSR

Castor est mis de côté, on ne concentre que sur les sorties de SVEN
"""


def load_clean_data(path_to_bem, path_to_sven) :
    """
    Reprendre le code d'import des données pour la stratégie globale --> les grilles
    radiales et azimutales sont fixés
     
    On impose les conventions suivantes : 
        - les données doivent être stockées dans des .xlslx
        - Une colonne correspond à un attribut

    Retourne un data frame pandas dans lequel toutes les données sont stockées, à savoir :
        - r
        - theta
        - (cos(theta), sin(theta))
        - yaw
        - TSR
        - v_eff
        - alpha
        - a
        - phi
        - Fn
        - Ft
    """

    sufs = [path_to_bem.suffix, path_to_sven.suffix]

    for suf in sufs :
        if suf == '.xlsx' :
            df_bem  = pd.read_excel(path_to_bem)
            df_sven = pd.read_excel(path_to_sven)
        if suf == '' :
            raise ImportError(f'Le chemin doit pointer directement sur le fichier à importer. Réessayez en pointant directement vers ce fichier.')
        else : 
            raise ImportError(f'Le fichier pointé doit être un .xlsx')
        
    df_merged = df_bem.merge(df_sven, on = ['r', 'theta'])


    # Encodage trigonométrique de l'azimuth (faudra en discuter)
    radians = np.radians(df_merged['theta'])
    df_merged['cos_theta'] = np.cos(radians)
    df_merged['sin_theta'] = np.sin(radians) 

    df_merged = df_merged.drop(columns = ['r', 'theta'])

    return df_merged

def get_splits(df, test_size = 0.2, save_dir = None) :
    """
    C'est plus simple dans ce cas : il n'y a qu'une stratégie !

    On drop les rayons et les azimuths. 

    features : 
        - Yaw
        - TSR

    Label (avant le choix complet de la stratégie) :
        - Fn
        - Ft
        - Variables intermédiaires
    """

    if 'yaw' not in list(df.keys()) or 'TSR' not in list(df.keys()):
        raise KeyError("Absence de yaw ou de TSR dans les données.")
    
    full_data = pd.concat([df['yaw'], df['TSR']], axis=1)
    train_df, val_df = train_test_split(full_data, test_size = test_size, random_state = 42)
    train_full = df[df['yaw'].isin(train_df)].copy()
    val_full = df[df['yaw'].isin(val_df)].copy()

    if save_dir != None : 
        if not exists(save_dir):
            os.mkdir(save_dir)
        train_full.to_csv(os.path.join(save_dir, "train_global.csv"), index = False)
        val_full.to_csv(os.path.join(save_dir, "val_global.csv"), index = False)
        print(f"Données écrites à {save_dir}")

    return train_full, val_full


def format_data(df, res, inter, is_train = True):
    """
    On conserve l'idée des plusieurs stratégies (inter,direct) x (bem,no_bem)
    """
    # 1. Sélection des variables intermédiaires
    if inter == 'f':
        cols_sven, cols_bem = ['Fn_SVEN', 'Ft_SVEN'], ['Fn_BEM', 'Ft_BEM']
    elif inter == 'v':
        cols_sven, cols_bem = ['V_eff_SVEN', 'alpha_SVEN'], ['V_eff_BEM', 'alpha_BEM']
    elif inter == 'u':
        cols_sven, cols_bem = ['a_SVEN', 'phi_SVEN'], ['a_BEM', 'phi_BEM']
    else:
        raise ValueError(f"Intermédiaire '{inter}' non reconnu.")

    # 2. Sélection des données (approche résiduelle ou directe)
    if res == 1 :
        Y_np = df[cols_sven].values - df[cols_bem].values
        X_np = df[['yaw', 'TSR']].values
    else :
        Y_np = df[cols_sven].values
        X_np = df[['yaw', 'TSR']]

    # 3. Normalisation

    model_name = f"global_{res}_{inter}"
    os.makedirs("scalers", exist_ok=True)
    path_x = f"scalers/scaler_X_{model_name}.pkl"
    path_y = f"scalers/scaler_Y_{model_name}.pkl"    

    if is_train:
        # On apprend la normalisation sur le train et on sauvegarde
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_np)
        Y_scaled = scaler_Y.fit_transform(Y_np)
        
        with open(path_x, 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(path_y, 'wb') as f:
            pickle.dump(scaler_Y, f)
    else:
        # On charge la normalisation existante pour le test
        with open(path_x, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(path_y, 'rb') as f:
            scaler_Y = pickle.load(f)
            
        X_scaled = scaler_X.transform(X_np)
        Y_scaled = scaler_Y.transform(Y_np)    


    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(Y_scaled, dtype=torch.float32)

def MB_build(train_full, val_full, batch_size, res:int, inter:str, is_train = True, seed = 42) : 
    """
    Création des mini-batchs. Appelle la fonction format_data et créer les batchs.
    Ajoute une dimension aux tenseurs pour qu'ils puissent rentrer dans le NN et envoie
    sur GPU si disponible sur la machine.
    """
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## 1. Construire train_dataloader
    feat_train, label_train = format_data(train_full, res, inter, is_train)
    feat_train = feat_train.unsqueeze(1).to(device); label_train = label_train.unsqueeze(1).to(device)
    train_dataloader = DataLoader(TensorDataset(feat_train, label_train),
                                  batch_size = batch_size, pin_memory = torch.cuda.is_available(),
                                  shuffle = True, generator = seed)
    k = 0
    for t in train_dataloader : 
        k+=1
    print("Nombre d'éléments dans le train_dataloader : ", k)

    ## 2. Construire val_dataloader
    feat_val, label_val = format_data(val_full, res, inter, is_train)
    feat_val = feat_val.unsqueeze(1).to(device); label_val = label_val.unsqueeze(1).to(device)
    val_dataloader = DataLoader(TensorDataset(feat_val, label_val),
                                  batch_size = batch_size, pin_memory = torch.cuda.is_available(),
                                  shuffle = True, generator = seed)
    k = 0
    for t in train_dataloader : 
        k+=1
    print("Nombre d'éléments dans le train_dataloader : ", k)

    return train_dataloader, val_dataloader