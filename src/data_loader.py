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

Castor est mis de côté, on ne concentre que sur les sorties de SVEN.

Liste exhaustive des paramètres :
    - entree --> Choix de la stratégie de correction
                    1. 'L'  == Stratégie locale : le modèle considère --en plus du yaw et du TSR-- les rayons et les azimuts
                    2. 'GR' == Stratégie semi-locale : le modèle prend juste le rayon en plus mais pas l'azimut
                    3. 'GA' == Stratégie semi-locale : le modèle prend  l'azimut mais pas le rayon
                    4. 'G'  == Stratégie purement gloable : seul le yaw et le TSR sont considérés

    - res    --> Choisir si il faut optimiser 
                    1. |FN_BEM-FT_SVEN| (res == 0)
                    2. FN_SVEN          (res == 1)

    - inter  --> Choisir la variables qui sera optimisé
                    1. 'f' == Apprentissage direct sur sur les forces de SVEN
                    2. 'u' == Apprentissage sur (V_eff, aoa)


"""


def load_clean_data(path_forces="data/raw/fichier_forces.csv", path_vitesses="data/raw/fichier_vitesses.csv"):
    """
    Charge les fichiers CSV de forces et de vitesses, les fusionne sur 
    les coordonnées communes, et ajoute l'encodage cyclique.
    """
    df_f = pd.read_csv(path_forces)
    df_v = pd.read_csv(path_vitesses)

    # Fusion sur les colonnes communes
    df_merged = df_f.merge(df_v, on=['yaw', 'r', 'theta'])

    # Encodage trigonométrique de l'azimut
    radians = np.radians(df_merged['theta'])
    df_merged['cos_theta'] = np.cos(radians)
    df_merged['sin_theta'] = np.sin(radians) 

    return df_merged

def get_splits(df, entree, seed = 42, test_size = 0.2, save_dir = None) :
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
    
    if entree == 'L':
        # Split point par point classique
        train_full, val_full = train_test_split(df, test_size = test_size, random_state = seed)
        
    elif entree == 'GR':
        # Split sur les Azimuts (on garde des profils de pales entiers)
        thetas_uniques = df['theta'].unique()
        train_th, test_th = train_test_split(thetas_uniques, test_size = test_size, random_state = 42)
        train_full = df[df['theta'].isin(train_th)].copy()
        val_full = df[df['theta'].isin(test_th)].copy()
        
    elif entree == 'GA':
        # Split sur les Rayons (on garde des anneaux de rotor entiers)
        rayons_uniques = df['r'].unique()
        train_r, test_r = train_test_split(rayons_uniques, test_size = test_size, random_state = seed)
        train_full = df[df['r'].isin(train_r)].copy()
        val_full = df[df['r'].isin(test_r)].copy()
    
    elif entree == 'G' :
    # On split sur les Yaws 
        yaws_uniques = df['yaw'].unique()
        train_yaw, test_yaw = train_test_split(yaws_uniques, test_size=test_size, random_state=seed)
        train_full = df[df['yaw'].isin(train_yaw)].copy()
        val_full = df[df['yaw'].isin(test_yaw)].copy()

    else :
        raise ValueError(f'\n{entree} est invalide.\n')

    if save_dir != None : 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        train_full.to_csv(os.path.join(save_dir, "train_global.csv"), index = False)
        val_full.to_csv(os.path.join(save_dir, "val_global.csv"), index = False)
        print(f"Données écrites à {save_dir}")

    return train_full, val_full


def format_data(df, entree, res, inter, is_train, device = 'cpu'):
    """
    On conserve l'idée des plusieurs stratégies (inter,direct) x (bem,no_bem)
    """
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
        if res_str == '1': # Stratégie résiduelle
            Y_np = df[cols_sven].values - df[cols_bem].values
            X_np = df[['r', 'cos_theta', 'sin_theta', 'yaw'] + cols_bem].values
        else: # Stratégie directe
            Y_np = df[cols_sven].values
            X_np = df[['r', 'cos_theta', 'sin_theta', 'yaw']].values

    elif entree == 'GR':
        # On groupe par angle ET par condition de vent
        grouped = df.groupby(['theta', 'yaw'])
        X_list, Y_list = [], []
        for (th, y_val), group in grouped:
            group = group.sort_values('r')
            y_sven = group[cols_sven].values.flatten()
            y_bem = group[cols_bem].values.flatten()
            
            # Extraction d'un seul scalaire par groupe
            c_th = group['cos_theta'].iloc[0]
            s_th = group['sin_theta'].iloc[0]
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                # Tableau 1D des entrées + données BEM
                X_val = np.concatenate(([c_th, s_th, y_val], y_bem))
            else:
                Y_val = y_sven
                X_val = np.array([c_th, s_th, y_val])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
        X_np = np.array(X_list)
        Y_np = np.array(Y_list)

    elif entree == 'GA':
        # On groupe par rayon ET par condition de vent
        grouped = df.groupby(['r', 'yaw'])
        X_list, Y_list = [], []
        for (r_val, y_val), group in grouped:
            group = group.sort_values('theta')
            y_sven = group[cols_sven].values.flatten()
            y_bem = group[cols_bem].values.flatten()

            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.concatenate(([r_val, y_val], y_bem))
            else:
                Y_val = y_sven
                X_val = np.array([r_val, y_val])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
        X_np = np.array(X_list)
        Y_np = np.array(Y_list)

    elif entree == 'G': 
        grouped = df.groupby('yaw')
        X_list, Y_list = [], []
        for y_val, group in grouped:
            # Tri strict pour que le vecteur soit toujours dans le même ordre (theta puis r)
            group = group.sort_values(['theta', 'r'])
            y_sven = group[cols_sven].values.flatten()
            y_bem = group[cols_bem].values.flatten()
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.array([y_val]) # Entrée = juste le Yaw
            else:
                Y_val = y_sven
                X_val = np.array([y_val])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
        X_np = np.array(X_list)
        Y_np = np.array(Y_list)

    # 3. Normalisation
    model_name = f"{entree}_{res_str}_{inter}"
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


    return torch.tensor(X_scaled, dtype=torch.float32, device = device), torch.tensor(Y_scaled, dtype=torch.float32, device = device)

# jolie fonction mais vaut mieux utiliser DataLoader directement dans les autres fichiers à chaque fois que t'en as besoin. 
def MB_build(df, batch_size, entree:str, res:int, inter:str, test_size = 0.2, is_train = True, seed = 42, save_dir = None, device = 'cpu') : 
    """
    Création des mini-batchs. Appelle la fonction format_data et créer les batchs.
    Ajoute une dimension aux tenseurs pour qu'ils puissent rentrer dans le NN et envoie
    sur GPU si disponible sur la machine.
    """
    
    ## 1. Récupérer les bon splits
    train_full, val_full = get_splits(df, entree, seed, test_size, save_dir)

    ## 2. Construire train_dataloader
    feat_train, label_train = format_data(train_full, entree = entree, res = res, inter = inter, is_train = is_train, device = device)
    feat_train = feat_train.unsqueeze(1); label_train = label_train.unsqueeze(1)
    train_dataloader = DataLoader(TensorDataset(feat_train, label_train),
                                  batch_size = batch_size,
                                  shuffle = True)
    k = 0
    for t in train_dataloader : 
        k+=1
    print("Nombre d'éléments dans le train_dataloader : ", k)

    ## 3. Construire val_dataloader
    feat_val, label_val = format_data(val_full, entree = entree, res = res, inter = inter, is_train = is_train)
    feat_val = feat_val.unsqueeze(1).to(device); label_val = label_val.unsqueeze(1).to(device)
    val_dataloader = DataLoader(TensorDataset(feat_val, label_val),
                                  batch_size = batch_size,
                                  shuffle = True)
    k = 0
    for t in val_dataloader : 
        k+=1
    print("Nombre d'éléments dans le train_dataloader : ", k)

    return train_dataloader, val_dataloader