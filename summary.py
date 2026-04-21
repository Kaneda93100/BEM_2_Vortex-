import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

# Imports internes
from src.data_loader import format_data
from src.models import TurbineMLP
from src.evaluate import reconstruct_predictions
from src.physics import convert_u_to_v, convert_v_to_f

def retrain_and_plot(model_name, rank_label, img_dir="images/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ▶ Analyse visuelle détaillée : {model_name} ({rank_label}) [Matériel : {device}]")
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        entree, residuelle, inter = model_name.split('_')
        with open(f"hyperparametres/{model_name}.json", "r") as f:
            hparams = json.load(f)
    except Exception as e:
        print(f" Erreur : {e}")
        return

    df_train = pd.read_csv(f"data/processed/train_{entree}.csv")
    df_test = pd.read_csv(f"data/processed/test_{entree}.csv")
    
    # NORMALISATION
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
    
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_full_train.numpy(), Y_full_train.numpy(), test_size=0.2, random_state=42)
    
    X_tr, Y_tr = torch.tensor(X_tr).to(device), torch.tensor(Y_tr).to(device)
    X_val, Y_val = torch.tensor(X_val).to(device), torch.tensor(Y_val).to(device)
    X_full_train_dev = X_full_train.to(device)
    X_test_dev = X_test.to(device)
    
    model = TurbineMLP(X_tr.shape[1], Y_tr.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    patience = 400 # Patience augmentée pour la synthèse visuelle
    best_val_loss, best_weights = float('inf'), None
    
    for epoch in range(2000):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_tr), Y_tr)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()
        history['epoch'].append(epoch); history['train_loss'].append(loss.item()); history['val_loss'].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; epochs_no_improve = 0; best_weights = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience: break
            
    model.load_state_dict(best_weights); model.eval()
    with torch.no_grad():
        preds_tr = model(X_full_train_dev).cpu().numpy()
        preds_te = model(X_test_dev).cpu().numpy()
        
    with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
    preds_tr = scaler_Y.inverse_transform(preds_tr)
    preds_te = scaler_Y.inverse_transform(preds_te)

    df_res_tr = reconstruct_predictions(df_train, preds_tr, entree, residuelle, inter)
    df_res_te = reconstruct_predictions(df_test, preds_te, entree, residuelle, inter)
    
    for df in [df_res_tr, df_res_te]:
        if inter == 'u':
            df['V_eff_pred'], df['alpha_pred'] = convert_u_to_v(df['a_pred'].values, df['phi_pred'].values, df['r'].values)
            df['Fn_pred'], df['Ft_pred'] = convert_v_to_f(df['V_eff_pred'].values, df['alpha_pred'].values, df['r'].values)
        elif inter == 'v':
            df['Fn_pred'], df['Ft_pred'] = convert_v_to_f(df['V_eff_pred'].values, df['alpha_pred'].values, df['r'].values)

    metrics = {}
    target_cols = ['Fn', 'Ft']
    if inter == 'v': target_cols += ['V_eff', 'alpha']
    if inter == 'u': target_cols += ['a', 'phi']
    for col in target_cols:
        s, p = df_res_te[f'{col}_SVEN'].values, df_res_te[f'{col}_pred'].values
        metrics[f'{col}_abs'] = np.sqrt(np.mean((p - s)**2))
        metrics[f'{col}_rel'] = (metrics[f'{col}_abs'] / np.mean(np.abs(s)) * 100) if np.mean(np.abs(s)) != 0 else 0

    _draw_plot(model_name, rank_label, history, df_res_tr, df_res_te, inter, img_dir, metrics)

def _draw_plot(model_name, rank_label, history, df_res_tr, df_res_te, inter, img_dir, metrics):
    n_cols = 4 if inter in ['u', 'v'] else 2
    n_rows = 3
    
    var_map = {
        'v': ('V_eff', '$V_{eff}$ [m/s]', 'alpha', '$\\alpha$ [deg]'),
        'u': ('a', 'Induction $a$', 'phi', 'Angle flux $\\phi$ [deg]')
    }
    
    # 6 Rayons
    rayons = np.sort(df_res_te['r'].unique())
    sections_r = rayons[np.linspace(0, len(rayons)-1, 6, dtype=int)] if len(rayons) > 6 else rayons
    
    # 3 Azimuts cibles : 0, 120, 240
    azimuts = np.sort(df_res_te['theta'].unique())
    sections_th = np.unique([azimuts[np.abs(azimuts - t).argmin()] for t in [0, 120, 240]])
    
    # --- PALETTE DE COULEURS MISE À JOUR ---
    # Rayons (6) : Palette qualitative standard
    colors_r = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # Azimuts (3) : Rouge, Vert, Bleu (Haut contraste, pas de noir)
    colors_th = ['#e41a1c', '#4daf4a', '#377eb8']

    fig = plt.figure(figsize=(5 * n_cols, 15))
    fig.suptitle(f"Synthèse Modèle : {model_name} ({rank_label})", fontsize=22, fontweight='bold', y=0.98)
    
    # 1. Convergence
    ax_h = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols)
    ax_h.plot(history['epoch'], history['train_loss'], color='blue', label='Train MSE')
    ax_h.plot(history['epoch'], history['val_loss'], color='red', linestyle='--', label='Val MSE')
    ax_h.set_yscale('log'); ax_h.set_title("Convergence", fontsize=14); ax_h.grid(True, alpha=0.3); ax_h.legend()

    plot_cols = [('Fn', 'Force Normale'), ('Ft', 'Force Tangentielle')]
    if inter in ['v', 'u']:
        plot_cols += [(var_map[inter][0], var_map[inter][1]), (var_map[inter][2], var_map[inter][3])]

    axes_th, axes_r = [], []
    for j, (key, label) in enumerate(plot_cols):
        ax_t = plt.subplot2grid((n_rows, n_cols), (1, j))
        ax_t.set_title(f"{label}\nRMSE: {metrics[key+'_abs']:.2f} ({metrics[key+'_rel']:.1f}%)", fontsize=11, fontweight='bold')
        ax_t.set_xlabel("Azimut θ [deg]"); ax_t.grid(True, alpha=0.2)
        axes_th.append(ax_t)
        
        ax_rad = plt.subplot2grid((n_rows, n_cols), (2, j))
        ax_rad.set_title(f"Profil spatial: {label}", fontsize=11, fontweight='bold')
        ax_rad.set_xlabel("Rayon r [m]"); ax_rad.grid(True, alpha=0.2)
        axes_r.append(ax_rad)

    # Ligne 2 (vs Theta)
    for i, r_val in enumerate(sections_r):
        dtr = df_res_tr[df_res_tr['r'] == r_val].sort_values('theta')
        dte = df_res_te[df_res_te['r'] == r_val].sort_values('theta')
        c = colors_r[i]
        for j, (key, _) in enumerate(plot_cols):
            axes_th[j].plot(dtr['theta'], dtr[f'{key}_SVEN'], color=c, marker='o', ls='none', ms=4, alpha=0.3)
            axes_th[j].plot(dte['theta'], dte[f'{key}_SVEN'], color=c, marker='*', ls='none', ms=7, alpha=0.7)
            full_p = pd.concat([dtr, dte]).sort_values('theta')
            axes_th[j].plot(full_p['theta'], full_p[f'{key}_pred'], color=c, ls='-', lw=2.5)

    # Ligne 3 (vs r)
    for i, th_val in enumerate(sections_th):
        dtr = df_res_tr[df_res_tr['theta'] == th_val].sort_values('r')
        dte = df_res_te[df_res_te['theta'] == th_val].sort_values('r')
        c = colors_th[i]
        for j, (key, _) in enumerate(plot_cols):
            axes_r[j].plot(dtr['r'], dtr[f'{key}_SVEN'], color=c, marker='o', ls='none', ms=4, alpha=0.4)
            axes_r[j].plot(dte['r'], dte[f'{key}_SVEN'], color=c, marker='*', ls='none', ms=7, alpha=0.8)
            full_p = pd.concat([dtr, dte]).sort_values('r')
            axes_r[j].plot(full_p['r'], full_p[f'{key}_pred'], color=c, ls='-', lw=2.5)

    # --- LÉGENDES ---
    rad_h = [Line2D([0], [0], color=colors_r[i], lw=3) for i in range(len(sections_r))]
    axes_th[0].legend(rad_h, [f"r={r:.2f}m" for r in sections_r], loc='upper center', 
                      bbox_to_anchor=(0.5, -0.22), ncol=3, title="Rayons", frameon=False)
    
    style_h = [Line2D([0], [0], color='gray', marker='o', ls='none'), 
               Line2D([0], [0], color='gray', marker='*', ls='none'), 
               Line2D([0], [0], color='black', ls='-', lw=2)]
    axes_th[1].legend(style_h, ["SVEN Train", "SVEN Test", "Prédiction ML"], loc='upper center', 
                      bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    
    th_h = [Line2D([0], [0], color=colors_th[i], lw=3) for i in range(len(sections_th))]
    axes_r[0].legend(th_h, [f"θ={th:.1f}°" for th in sections_th], loc='upper center', 
                     bbox_to_anchor=(0.5, -0.22), ncol=3, title="Azimuts", frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(img_dir, f"summary_{rank_label.replace(' ','_')}_{model_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_table(perf_dir="performance/", N_tab=2, N_fig=1):
    recap_file = os.path.join(perf_dir, "recap_scores_globaux.csv")
    if not os.path.exists(recap_file): return
    df_recap = pd.read_csv(recap_file)
    n = len(df_recap)
    
    if n <= 3 * N_tab:
        idx_list, labels = list(range(n)), [f"Rang {i+1}" for i in range(n)]
    else:
        idx_list = list(range(N_tab)) + list(range((n-N_tab)//2, (n-N_tab)//2 + N_tab)) + list(range(n-N_tab, n))
        labels = [f"TOP {i+1}" for i in range(N_tab)] + [f"MEDIAN {i+1}" for i in range(N_tab)] + [f"PIRE {N_tab-i}" for i in range(N_tab)]

    df_display = df_recap.iloc[idx_list].copy()
    df_display.insert(0, 'Rang', labels)
    
    print("\n CLASSEMENT FINAL")
    print("="*105)
    print(df_display.round(2).to_string(index=False))
    print("="*105)

    targets = {0: "TOP 1", n//2: "MEDIAN", n-1: "PIRE 1"}
    for idx, lbl in targets.items():
        retrain_and_plot(df_recap.iloc[idx]['Modele'], lbl)

generate_summary_table()