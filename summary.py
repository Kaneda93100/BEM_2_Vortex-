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
from src.models import TurbineMLP, EnsemblePointNet
from src.evaluate import reconstruct_predictions
from src.physics import compute_cp, convert_v_to_f

def get_marker_logic(df_train, yaw_val, total_points_per_yaw):
    """Détermine la forme du point SVEN selon le pourcentage de présence au training."""
    count = len(df_train[df_train['yaw'] == yaw_val])
    ratio = count / total_points_per_yaw if total_points_per_yaw > 0 else 0
    
    if ratio < 0.70: return '*', 11  # Étoile (Sous-représenté)
    elif ratio <= 0.90: return 's', 6   # Carré (Standard)
    else: return 'o', 6   # Rond (Sur-représenté)

def generate_comparison_summary():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Génération de la synthèse du duel G vs P sur {device}...")
    
    recap_path = "performance/recap_scores_globaux.csv"
    if not os.path.exists(recap_path):
        print("Erreur : recap_scores_globaux.csv introuvable. Lancez main.py d'abord.")
        return
    
    df_recap = pd.read_csv(recap_path)
    score_col = "Score_Total_%" if "Score_Total_%" in df_recap.columns else "Score_Global_%"
    
    # On exclut la baseline
    df_ml = df_recap[~df_recap['Modele'].str.contains('Baseline')].copy()
    
    df_raw = load_clean_data()
    points_per_yaw = df_raw.groupby('yaw').size().iloc[0] 
    
    # On doit retrouver le MÊME split que lors de l'entraînement
    df_train, df_test = get_splits(df_raw, seed=42, test_size=0.2)
    
    os.makedirs("images/rendements", exist_ok=True)
    
    # 1. Sélection des champions (Le meilleur G et le meilleur P)
    best_models = []
    for e in ['G', 'P']:
        df_fam = df_ml[df_ml['Modele'].str.startswith(f"{e}_")].sort_values(by=score_col)
        if not df_fam.empty:
            best_models.append(df_fam.iloc[0])
            
    if len(best_models) < 2:
        print("Il manque des modèles G ou P dans le fichier de résultats pour faire le duel.")
        return

    # 2. Préparation des figures
    fig, (ax_loss, ax_cp) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#1f77b4', '#d62728'] # Bleu pour G, Rouge pour P
    sven_data = None
    
    for idx, row in enumerate(best_models):
        model_name = row['Modele']
        score_forces = row[score_col]
        print(f"   Reconstruction de {model_name}...")
        
        entree, residuelle, inter = model_name.split('_')
        with open(f"hyperparametres/{model_name}.json", "r") as f:
            hp = json.load(f)
            
        # Formatage avec gestion du device
        X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
        X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
        
        # --- Instanciation des Modèles ---
        if entree == 'P':
            num_points = 1296
            num_yaws_train = len(X_train) // num_points
            num_yaws_test = len(X_test) // num_points
            
            # Transformation 3D
            X_train_3D = X_train.view(num_points, num_yaws_train, -1)
            Y_train_3D = Y_train.view(num_points, num_yaws_train, -1)
            X_test_3D = X_test.view(num_points, num_yaws_test, -1)
            
            model = EnsemblePointNet(num_points, X_train_3D.shape[2], Y_train_3D.shape[2], 
                                     hp['n_layers'], hp['n_neurons'], hp['dropout_rate']).to(device)
            X_tr_final, Y_tr_final = X_train_3D, Y_train_3D
        else:
            model = TurbineMLP(X_train.shape[1], Y_train.shape[1], 
                               hp['n_layers'], hp['n_neurons'], hp['dropout_rate']).to(device)
            X_tr_final, Y_tr_final = X_train, Y_train

        optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
        criterion = nn.MSELoss()
        
        # --- RE-ENTRAÎNEMENT RAPIDE POUR TRACER LA LOSS ---
        history_loss = []
        model.train()
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = criterion(model(X_tr_final), Y_tr_final)
            loss.backward()
            optimizer.step()
            history_loss.append(loss.item())
            
        ax_loss.plot(range(1000), history_loss, label=f"({entree}) {model_name} (Err: {score_forces:.1f}%)", color=colors[idx])
        
        # --- PRÉDICTION SUR TOUT LE ROTOR ---
        model.eval()
        with torch.no_grad():
            if entree == 'P':
                preds_tr_raw = model(X_train_3D).view(-1, Y_train_3D.shape[2]).cpu().numpy()
                preds_te_raw = model(X_test_3D).view(-1, Y_train_3D.shape[2]).cpu().numpy()
            else:
                preds_tr_raw = model(X_train).cpu().numpy()
                preds_te_raw = model(X_test).cpu().numpy()
        
        with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f:
            scaler_Y = pickle.load(f)
            
        preds_tr = scaler_Y.inverse_transform(preds_tr_raw)
        preds_te = scaler_Y.inverse_transform(preds_te_raw)
        
        df_res_tr = reconstruct_predictions(df_train, preds_tr, entree, residuelle, inter)
        df_res_te = reconstruct_predictions(df_test, preds_te, entree, residuelle, inter)
        
        # Fusion pour reformer la carte aérodynamique complète
        df_res_full = pd.concat([df_res_tr, df_res_te]).sort_values(by=['yaw', 'r', 'theta'])
        
        if inter == 'v':
            df_res_full['Fn_pred'], df_res_full['Ft_pred'] = convert_v_to_f(
                df_res_full['V_eff_pred'].values, df_res_full['alpha_pred'].values, df_res_full['r'].values
            )
        
        # --- CALCUL DES CP ---
        df_cp_ml = compute_cp(df_res_full, 'Fn_pred', 'Ft_pred').sort_values('yaw')
        df_cp_sven = compute_cp(df_res_full, 'Fn_SVEN', 'Ft_SVEN').sort_values('yaw')
        sven_data = df_cp_sven 
        
        # Tracé de la courbe ML
        ax_cp.plot(df_cp_ml['yaw'], df_cp_ml['Cp_pred'], 
                   label=f"Modèle {entree} ({model_name})", 
                   color=colors[idx], lw=2.5)

    # --- TRACÉ DES VRAIES VALEURS SVEN ---
    for _, row in sven_data.iterrows():
        m, s = get_marker_logic(df_train, row['yaw'], points_per_yaw)
        ax_cp.plot(row['yaw'], row['Cp_SVEN'], marker=m, markersize=s, 
                   color='black', linestyle='None', alpha=0.7)

    # --- COSMÉTIQUE GRAPHIQUE ---
    ax_loss.set_title(r"Convergence de l'entraînement (MSE)", fontsize=13)
    ax_loss.set_xlabel("Époques")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc='upper right')

    ax_cp.set_title(r"Duel des Rendements ($C_P$) : Global vs Point-wise", fontsize=13)
    ax_cp.set_xlabel(r"Angle de Yaw ($^{\circ}$)")
    ax_cp.set_ylabel(r"Coefficient de Puissance ($C_P$)")
    ax_cp.grid(True, linestyle='--', alpha=0.5)
    
    custom_lines = [
        Line2D([0], [0], marker='o', color='black', label='Vérité (>90% données)', ls='None'),
        Line2D([0], [0], marker='s', color='black', label='Vérité (70-90% données)', ls='None'),
        Line2D([0], [0], marker='*', color='black', markersize=10, label='Vérité (<70% données)', ls='None')
    ]
    ax_cp.legend(handles=ax_cp.get_legend_handles_labels()[0] + custom_lines, loc='lower left')

    plt.suptitle("Duel des Architectures : Modèle Global (G) vs Ensemble Local (P)", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    file_path = "images/rendements/synthese_G_vs_P.png"
    plt.savefig(file_path, dpi=150)
    print(f" Image enregistrée : {file_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_summary()