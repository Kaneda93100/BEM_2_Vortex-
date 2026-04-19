import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os

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

def format_data(df, entree, residuelle, inter):
    """
    Transforme les DataFrames Pandas en Tenseurs PyTorch.
    Gère l'aplatissement pour les méthodes globales.
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

    # Sécurité : forcer residuelle en string
    res_str = str(residuelle)

    # ==========================================
    # APPROCHE LOCALE (L)
    # ==========================================
    if entree == 'L':
        if res_str == '1':
            Y = df[cols_sven].values - df[cols_bem].values
            X = df[['r', 'cos_theta', 'sin_theta'] + cols_bem].values
        else:
            Y = df[cols_sven].values
            X = df[['r', 'cos_theta', 'sin_theta']].values
            
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    # ==========================================
    # APPROCHE GLOBALE RAYONS FIXÉS (GR)
    # ==========================================
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
            else: # Correction appliquée ici
                Y_val = y_sven
                X_val = np.array([cos_th, sin_th])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
            
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(Y_list), dtype=torch.float32)

    # ==========================================
    # APPROCHE GLOBALE AZIMUTS FIXÉS (GA)
    # ==========================================
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
            else: # Correction appliquée ici
                Y_val = y_sven
                X_val = np.array([r_val])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
            
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(Y_list), dtype=torch.float32)