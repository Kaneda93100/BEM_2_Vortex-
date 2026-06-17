import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle
from sklearn.preprocessing import StandardScaler

def load_clean_data(path_forces="data/raw/fichier_forces.csv", path_vitesses="data/raw/fichier_vitesses.csv"):
    df_f = pd.read_csv(path_forces)
    df_v = pd.read_csv(path_vitesses)
    
    merge_keys = ['yaw', 'r', 'theta']
    if 'TSR' in df_f.columns and 'TSR' in df_v.columns:
        merge_keys.append('TSR')
        
    return df_f.merge(df_v, on=merge_keys)

def get_splits(df, seed=42, test_size=0.2, save_dir=None):
    yaws_uniques = df['yaw'].unique()
    train_yaw, test_yaw = train_test_split(yaws_uniques, test_size=test_size, random_state=seed)
    
    train_df = df[df['yaw'].isin(train_yaw)].copy()
    test_df = df[df['yaw'].isin(test_yaw)].copy()
    
    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        print(f" [OK] Fichiers créés : train.csv et test.csv dans {save_dir}")

    return train_df, test_df

def get_D_tensor(df, entree, device='cpu'):
    """ Calcule et formate le tenseur de pression dynamique D """
    from core.physics import compute_dynamic_pressure_D
    df_calc = df.copy()
    if 'TSR' not in df_calc.columns: 
        df_calc['TSR'] = 8.0
        
    df_calc['D'] = compute_dynamic_pressure_D(df_calc).clip(lower=1e-6)
    
    D_list = []
    grouped = df_calc.groupby(['yaw', 'TSR'])
    
    if entree == 'GV':
        for _, group in grouped:
            group = group.sort_values(['theta', 'r'])
            # Y a deux valeurs (Fn, Ft) par noeud, on duplique donc D pour correspondre
            D_flat = np.repeat(group['D'].values, 2)
            D_list.append(D_flat)
            
    elif entree == 'GM':
        r_uniques = np.sort(df_calc['r'].unique())
        theta_uniques = np.sort(df_calc['theta'].unique())
        for _, group in grouped:
            group = group.sort_values(['r', 'theta'])
            d_val = group['D'].values.reshape(len(r_uniques), len(theta_uniques))
            d_tensor_img = np.stack([d_val, d_val])
            D_list.append(d_tensor_img)
            
    return torch.tensor(np.array(D_list), dtype=torch.float32, device=device)

def format_data(df, entree, res, inter, is_train, device='cpu'):
    res_str = str(res)
    df_work = df.copy()
    if 'TSR' not in df_work.columns:
        df_work['TSR'] = 8.0
    
    if inter == 'f': 
        from core.physics import compute_dynamic_pressure_D
        D = compute_dynamic_pressure_D(df_work).clip(lower=1e-6)
        cols_sven, cols_bem = ['Fn_SVEN', 'Ft_SVEN'], ['Fn_BEM', 'Ft_BEM']
        
        for col in cols_sven + cols_bem:
            df_work[col] = df_work[col] / D
            
    elif inter == 'v': 
        cols_sven, cols_bem = ['V_eff_SVEN', 'alpha_SVEN'], ['V_eff_BEM', 'alpha_BEM']

    # --- Stratégie GV ---
    if entree == 'GV': 
        grouped = df_work.groupby(['yaw', 'TSR'])
        X_list, Y_list = [], []
        
        for (y_val, tsr_val), group in grouped:
            group = group.sort_values(['theta', 'r'])
            y_sven = group[cols_sven].values.flatten()
            y_bem = group[cols_bem].values.flatten()
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.concatenate(([y_val, tsr_val], y_bem)) 
            else:
                Y_val = y_sven
                X_val = np.array([y_val, tsr_val])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
        X_np, Y_np = np.array(X_list), np.array(Y_list)

    # --- Stratégie GM ---
    elif entree == 'GM':
        grouped = df_work.groupby(['yaw', 'TSR'])
        X_list, Y_list = [], []
        
        r_uniques = np.sort(df_work['r'].unique())
        theta_uniques = np.sort(df_work['theta'].unique())
        
        for (y_val, tsr_val), group in grouped:
            group = group.sort_values(['r', 'theta'])
            
            y_sven = group[cols_sven].values.reshape(len(r_uniques), len(theta_uniques), 2).transpose(2, 0, 1)
            y_bem = group[cols_bem].values.reshape(len(r_uniques), len(theta_uniques), 2).transpose(2, 0, 1)
            
            yaw_channel = np.full((len(r_uniques), len(theta_uniques)), y_val)
            tsr_channel = np.full((len(r_uniques), len(theta_uniques)), tsr_val)
            
            if res_str == '1':
                Y_val = y_sven - y_bem
                X_val = np.stack([y_bem[0], y_bem[1], yaw_channel, tsr_channel])
            else:
                Y_val = y_sven
                R_grid, Theta_grid = np.meshgrid(r_uniques, theta_uniques, indexing='ij')
                X_val = np.stack([yaw_channel, tsr_channel, R_grid, Theta_grid])
                
            X_list.append(X_val)
            Y_list.append(Y_val)
            
        X_np, Y_np = np.array(X_list), np.array(Y_list)

    else:
        raise ValueError(f"Stratégie '{entree}' non reconnue. Utilisez 'GV' ou 'GM'.")

    # ==========================================
    # NORMALISATION (SCALING)
    # ==========================================
    model_name = f"{entree}_{res_str}_{inter}"
    os.makedirs("training/scalers", exist_ok=True)
    path_x = f"training/scalers/scaler_X_{model_name}.pkl"
    path_y = f"training/scalers/scaler_Y_{model_name}.pkl"    

    original_shape_X = X_np.shape
    original_shape_Y = Y_np.shape
    
    if entree == 'GM':
        X_np = X_np.reshape(X_np.shape[0], -1)
        Y_np = Y_np.reshape(Y_np.shape[0], -1)

    if is_train:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_np)
        with open(path_x, 'wb') as f: 
            pickle.dump(scaler_X, f)
        

        if inter == 'v':
            scaler_Y = StandardScaler()
            Y_scaled = scaler_Y.fit_transform(Y_np)
            with open(path_y, 'wb') as f: 
                pickle.dump(scaler_Y, f)
        else:
            Y_scaled = Y_np
            
    else:
        with open(path_x, 'rb') as f: 
            scaler_X = pickle.load(f)
        X_scaled = scaler_X.transform(X_np)
        
        if inter == 'v':
            with open(path_y, 'rb') as f: 
                scaler_Y = pickle.load(f)
            Y_scaled = scaler_Y.transform(Y_np)
        else:
            Y_scaled = Y_np    

    if entree == 'GM':
        X_scaled = X_scaled.reshape(original_shape_X)
        Y_scaled = Y_scaled.reshape(original_shape_Y)

    return torch.tensor(X_scaled, dtype=torch.float32, device=device), torch.tensor(Y_scaled, dtype=torch.float32, device=device)