import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler

def load_clean_data(path_forces="data/raw/fichier_forces.csv", 
                    path_vitesses="data/raw/fichier_vitesses.csv", 
                    path_induction="data/raw/fichier_induction.csv"):
    """
    Lit et fusionne les données, retire Castor, et ajoute l'encodage 
    cyclique de l'azimut (cos_theta, sin_theta).
    """
    df_f = pd.read_csv(path_forces)
    df_v = pd.read_csv(path_vitesses)
    df_i = pd.read_csv(path_induction)

    # Fusion sur les coordonnées spatiales
    df_merged = df_f.merge(df_v, on=['r', 'theta']).merge(df_i, on=['r', 'theta'])
    
    # Retrait des données Castor (inutiles pour le ML)
    cols_to_drop = [col for col in df_merged.columns if 'Castor' in col]
    df_merged = df_merged.drop(columns=cols_to_drop)

    # NOUVEAU : Encodage cyclique de l'azimut pour éviter la discontinuité 359° -> 0°
    theta_rad = np.radians(df_merged['theta'])
    df_merged['cos_theta'] = np.cos(theta_rad)
    df_merged['sin_theta'] = np.sin(theta_rad)

    return df_merged

def get_splits(df, entree, save_dir="data/processed/"):
    """
    Crée les splits Train/Test en respectant la structure de la grille 
    selon la stratégie spatiale (L, GR, GA).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if entree == 'L':
        # Split point par point classique
        train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)
        
    elif entree == 'GR':
        # Split sur les Azimuts (on garde des profils de pales entiers)
        thetas_uniques = df['theta'].unique()
        train_th, test_th = train_test_split(thetas_uniques, test_size=0.30, random_state=42)
        train_df = df[df['theta'].isin(train_th)].copy()
        test_df = df[df['theta'].isin(test_th)].copy()
        
    elif entree == 'GA':
        # Split sur les Rayons (on garde des anneaux de rotor entiers)
        rayons_uniques = df['r'].unique()
        train_r, test_r = train_test_split(rayons_uniques, test_size=0.30, random_state=42)
        train_df = df[df['r'].isin(train_r)].copy()
        test_df = df[df['r'].isin(test_r)].copy()

    # Sauvegarde (utile pour le debug)
    train_df.to_csv(os.path.join(save_dir, f"train_{entree}.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, f"test_{entree}.csv"), index=False)

    return train_df, test_df

def format_data(df, entree, residuelle, inter, is_train=True):
    """
    Transforme les DataFrames Pandas en Tenseurs PyTorch.
    Gère l'aplatissement pour les méthodes globales et applique 
    une standardisation
    """
    # 1. Sélection des colonnes cibles
    if inter == 'f':
        cols_sven, cols_bem = ['Fn_SVEN', 'Ft_SVEN'], ['Fn_BEM', 'Ft_BEM']
    elif inter == 'v':
        cols_sven, cols_bem = ['V_eff_SVEN', 'alpha_SVEN'], ['V_eff_BEM', 'alpha_BEM']
    elif inter == 'u':
        cols_sven, cols_bem = ['a_SVEN', 'phi_SVEN'], ['a_BEM', 'phi_BEM']
    else:
        raise ValueError(f"Intermédiaire '{inter}' non reconnu.")

    res_str = str(residuelle)

    # 2. Extraction des matrices brutes selon l'approche spatiale
    if entree == 'L':
        if res_str == '1':
            Y_np = df[cols_sven].values - df[cols_bem].values
            X_np = df[['r', 'cos_theta', 'sin_theta'] + cols_bem].values
        else:
            Y_np = df[cols_sven].values
            X_np = df[['r', 'cos_theta', 'sin_theta']].values

    elif entree == 'GR':
        grouped = df.groupby('theta')
        X_list, Y_list = [], []
        for th, group in grouped:
            group = group.sort_values('r')
            y_sven = group[cols_sven].values.flatten()
            y_bem = group[cols_bem].values.flatten()
            cos_th = group['cos_theta'].iloc[0]
            sin_th = group['sin_theta'].iloc[0]
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.concatenate(([cos_th, sin_th], y_bem))
            else:
                Y_val = y_sven
                X_val = np.array([cos_th, sin_th])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
        X_np = np.array(X_list)
        Y_np = np.array(Y_list)

    elif entree == 'GA':
        grouped = df.groupby('r')
        X_list, Y_list = [], []
        for r_val, group in grouped:
            group = group.sort_values('theta')
            y_sven = group[cols_sven].values.flatten()
            y_bem = group[cols_bem].values.flatten()
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.concatenate(([r_val], y_bem))
            else:
                Y_val = y_sven
                X_val = np.array([r_val])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
        X_np = np.array(X_list)
        Y_np = np.array(Y_list)

    # ==========================================
    # 3. NORMALISATION (StandardScaler)
    # ==========================================
    model_name = f"{entree}_{residuelle}_{inter}"
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