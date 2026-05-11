import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.data_loader import load_clean_data, format_data, get_splits
from src.models import TurbineMLP
from src.evaluate import reconstruct_predictions
from src.physics import compute_cp, convert_v_to_f

def get_marker_logic(df_train, yaw_val, total_points_per_yaw):
    """Détermine la forme du point SVEN selon le pourcentage de présence au training."""
    count = len(df_train[df_train['yaw'] == yaw_val])
    ratio = count / total_points_per_yaw if total_points_per_yaw > 0 else 0
    
    if ratio < 0.70:
        return '*', 11  # Étoile (Sous-représenté)
    elif ratio <= 0.90:
        return 's', 6   # Carré (Standard)
    else:
        return 'o', 6   # Rond (Sur-représenté)

def generate_family_cp_summary():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Génération des synthèses par famille sur {device}...")
    
    recap_path = "performance/recap_scores_globaux.csv"
    if not os.path.exists(recap_path):
        print("Erreur : recap_scores_globaux.csv introuvable.")
        return
    
    df_recap = pd.read_csv(recap_path)
    df_ml = df_recap[~df_recap['Modele'].str.contains('Baseline')].copy()
    
    df_raw = load_clean_data()
    points_per_yaw = df_raw.groupby('yaw').size().iloc[0] 
    
    os.makedirs("images/rendements", exist_ok=True)
    
    familles = ['G', 'GR', 'L']

    for famille in familles:
        # Filtrer les modèles de cette famille exacte
        df_famille = df_ml[df_ml['Modele'].str.startswith(f"{famille}_")].sort_values(by="Score_Global_%")
        
        if df_famille.empty:
            continue
            
        print(f"\n Analyse de la famille {famille}...")
        
        # Sélection du meilleur et du pire
        best_model = df_famille.iloc[0]
        worst_model = df_famille.iloc[-1]
        
        modeles_a_tracer = [best_model]
        if best_model['Modele'] != worst_model['Modele']:
            modeles_a_tracer.append(worst_model)
            
        # Création de la figure avec 2 sous-graphiques (1x2)
        fig, (ax_loss, ax_cp) = plt.subplots(1, 2, figsize=(16, 7))
        colors = ['#1f77b4', '#d62728'] # Bleu pour le meilleur, Rouge pour le pire
        labels_legend_cp = []
        
        # Pour stocker les points SVEN à tracer une seule fois à la fin
        sven_data = None
        df_train_reference = None
        
        for idx, row in enumerate(modeles_a_tracer):
            model_name = row['Modele']
            score_forces = row['Score_Global_%']
            print(f"   Entraînement et prédiction de {model_name}...")
            
            entree, residuelle, inter = model_name.split('_')
            with open(f"hyperparametres/{model_name}.json", "r") as f:
                hp = json.load(f)
            
            df_train, df_test = get_splits(df_raw, entree=entree)
            if sven_data is None: df_train_reference = df_train
                
            X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=True)
            X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
            
            model = TurbineMLP(X_train.shape[1], Y_train.shape[1], hp['n_layers'], hp['n_neurons'], hp['dropout_rate']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
            criterion = nn.MSELoss()
            
            b_size = 1024 if entree == 'L' else (32 if entree in ['GR', 'GA'] else len(X_train))
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train.to(device), Y_train.to(device)), 
                batch_size=b_size, shuffle=True
            )
            
            # --- ENTRAÎNEMENT & RÉCUPÉRATION DE LA LOSS ---
            history_loss = []
            model.train()
            for epoch in range(1000):
                epoch_loss = 0.0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                history_loss.append(epoch_loss / len(train_loader))
                
            ax_loss.plot(range(1000), history_loss, label=f"{model_name} (Forces Err: {score_forces:.1f}%)", color=colors[idx])
            
            # --- PRÉDICTION SUR LE ROTOR COMPLET (Train + Test) ---
            model.eval()
            with torch.no_grad():
                preds_tr_raw = model(X_train.to(device)).cpu().numpy()
                preds_te_raw = model(X_test.to(device)).cpu().numpy()
            
            with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f:
                scaler_Y = pickle.load(f)
                
            preds_tr = scaler_Y.inverse_transform(preds_tr_raw)
            preds_te = scaler_Y.inverse_transform(preds_te_raw)
            
            df_res_tr = reconstruct_predictions(df_train, preds_tr, entree, residuelle, inter)
            df_res_te = reconstruct_predictions(df_test, preds_te, entree, residuelle, inter)
            
            # Concaténation pour reformer le rotor entier sans trous
            df_res_full = pd.concat([df_res_tr, df_res_te]).sort_values(by=['yaw', 'r', 'theta'])
            
            if inter == 'v':
                df_res_full['Fn_pred'], df_res_full['Ft_pred'] = convert_v_to_f(
                    df_res_full['V_eff_pred'].values, df_res_full['alpha_pred'].values, df_res_full['r'].values
                )
            
            # Calcul des Cp
            df_cp_ml = compute_cp(df_res_full, 'Fn_pred', 'Ft_pred').sort_values('yaw')
            df_cp_sven = compute_cp(df_res_full, 'Fn_SVEN', 'Ft_SVEN').sort_values('yaw')
            sven_data = df_cp_sven # On sauvegarde pour le tracé final
            
            # Erreur relative du Cp
            rmse_cp = np.sqrt(np.mean((df_cp_ml['Cp_pred'] - df_cp_sven['Cp_SVEN'])**2))
            mean_cp_sven = np.mean(np.abs(df_cp_sven['Cp_SVEN']))
            err_rel_cp = (rmse_cp / mean_cp_sven * 100) if mean_cp_sven != 0 else 0
            
            # Tracé de la courbe continue
            label_name = "Meilleur" if idx == 0 else "Pire"
            ax_cp.plot(df_cp_ml['yaw'], df_cp_ml['Cp_pred'], 
                     label=f"{label_name}: {model_name} (Err $C_P$: {err_rel_cp:.1f}%)", 
                     color=colors[idx], lw=2.5)

        # --- TRACÉ DES VRAIES VALEURS (SVEN) AVEC MARQUEURS ---
        for _, row in sven_data.iterrows():
            m, s = get_marker_logic(df_train_reference, row['yaw'], points_per_yaw)
            ax_cp.plot(row['yaw'], row['Cp_SVEN'], marker=m, markersize=s, 
                     color='black', linestyle='None', alpha=0.7)

        # --- COSMÉTIQUE GRAPHIQUE 1 : LOSS ---
        ax_loss.set_title(r"Convergence de l'entraînement (MSE)", fontsize=13)
        ax_loss.set_xlabel("Époques")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_yscale('log')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc='upper right')

        # --- COSMÉTIQUE GRAPHIQUE 2 : CP ---
        ax_cp.set_title(r"Comparaison des Rendements ($C_P$)", fontsize=13)
        ax_cp.set_xlabel(r"Angle de Yaw ($^{\circ}$)")
        ax_cp.set_ylabel(r"Coefficient de Puissance ($C_P$)")
        ax_cp.grid(True, linestyle='--', alpha=0.5)
        
        # Légende Combinée pour CP
        custom_lines = [
            Line2D([0], [0], marker='o', color='black', label='Vérité (>90% données)', ls='None'),
            Line2D([0], [0], marker='s', color='black', label='Vérité (70-90% données)', ls='None'),
            Line2D([0], [0], marker='*', color='black', markersize=10, label='Vérité (<70% données)', ls='None')
        ]
        ax_cp.legend(handles=ax_cp.get_legend_handles_labels()[0] + custom_lines, loc='lower left')

        # Titre Général
        plt.suptitle(f"Famille Stratégique : {famille} ", fontsize=16, fontweight='bold')
        
        # Sauvegarde
        plt.tight_layout()
        file_path = f"images/rendements/synthese_famille_{famille}.png"
        plt.savefig(file_path, dpi=150)
        print(f" Image enregistrée : {file_path}")
        plt.close()

if __name__ == "__main__":
    generate_family_cp_summary()