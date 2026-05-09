import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- CONFIGURATION ---
SEED = 42
SPAN = 2.25  # Longueur de la pale MEXICO
TARGET_PERCENTAGES = [0.20, 0.35, 0.50, 0.60, 0.75, 0.90]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from src.data_loader import format_data
from src.models import TurbineMLP
from src.evaluate import reconstruct_predictions
from src.physics import convert_v_to_f

def retrain_and_plot(model_name, rank_label, ml_score, baseline_scores, img_dir="images/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ▶ Analyse : {model_name} ({rank_label}) [Score ML: {ml_score:.2f}%]")
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        clean_model_name = model_name.strip()
        entree, residuelle, inter = clean_model_name.split('_')
        with open(f"hyperparametres/{clean_model_name}.json", "r") as f:
            hparams = json.load(f)
    except Exception as e:
        print(f" Erreur hyperparamètres : {e}")
        return

    df_train = pd.read_csv(f"data/processed/train_{entree}.csv")
    df_test = pd.read_csv(f"data/processed/test_{entree}.csv")
    
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
    
    X_tr, Y_tr = X_full_train.to(device), Y_full_train.to(device)
    X_test_dev, Y_test_dev = X_test.to(device), Y_test.to(device)
    
    model = TurbineMLP(X_tr.shape[1], Y_tr.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    history = {'epoch': [], 'train_loss': []}
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), Y_tr)
        loss.backward()
        optimizer.step()
        history['epoch'].append(epoch); history['train_loss'].append(loss.item())
            
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_dev), Y_test_dev).item()
        preds_tr = model(X_tr).cpu().numpy()
        preds_te = model(X_test_dev).cpu().numpy()
        
    with open(f"scalers/scaler_Y_{clean_model_name}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
    preds_tr = scaler_Y.inverse_transform(preds_tr)
    preds_te = scaler_Y.inverse_transform(preds_te)

    df_res_tr = reconstruct_predictions(df_train, preds_tr, entree, residuelle, inter)
    df_res_te = reconstruct_predictions(df_test, preds_te, entree, residuelle, inter)
    
    for df in [df_res_tr, df_res_te]:
        if inter == 'v':
            df['Fn_pred'], df['Ft_pred'] = convert_v_to_f(df['V_eff_pred'].values, df['alpha_pred'].values, df['r'].values)

    metrics = {}
    target_cols = ['Fn', 'Ft']
    if inter == 'v': target_cols += ['V_eff', 'alpha']
    
    for col in target_cols:
        s, p = df_res_te[f'{col}_SVEN'].values, df_res_te[f'{col}_pred'].values
        metrics[f'{col}_abs'] = np.sqrt(np.mean((p - s)**2))
        metrics[f'{col}_rel'] = (metrics[f'{col}_abs'] / np.mean(np.abs(s)) * 100) if np.mean(np.abs(s)) != 0 else 0

    _draw_plot(clean_model_name, rank_label, history, test_loss, df_res_tr, df_res_te, inter, img_dir, metrics, ml_score, baseline_scores)

def _draw_plot(model_name, rank_label, history, test_loss, df_res_tr, df_res_te, inter, img_dir, metrics, ml_score, baseline_scores):
    n_cols = 4 if inter == 'v' else 2
    n_rows = 3
    var_map = {'v': ('V_eff', '$V_{eff}$ [m/s]', 'alpha', '$\\alpha$ [deg]')}
    
    # --- SÉLECTION DES RAYONS CIBLES (Logic % du Span) ---
    all_radii = np.sort(df_res_te['r'].unique())
    target_radii_vals = [p * SPAN for p in TARGET_PERCENTAGES]
    sections_r = []
    for tr in target_radii_vals:
        idx = (np.abs(all_radii - tr)).argmin()
        sections_r.append(all_radii[idx])
    sections_r = np.unique(sections_r)
    
    azimuts = np.sort(df_res_te['theta'].unique())
    sections_th = np.unique([azimuts[np.abs(azimuts - t).argmin()] for t in [0, 120, 240]])
    
    colors_r = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors_th = ['#e41a1c', '#4daf4a', '#377eb8']

    fig = plt.figure(figsize=(5 * n_cols, 15))
    
    # --- HIÉRARCHIE DES TITRES ---
    fig.text(0.5, 0.98, f"Modèle : {model_name} ({rank_label})", ha='center', fontsize=22, fontweight='bold')
    fig.text(0.5, 0.955, f"SCORE ML GLOBAL : {ml_score:.2f}%", ha='center', fontsize=18, color='#28a745', fontweight='bold')
    
    b_info = "Baselines Physiques : "
    if baseline_scores['BEM']: b_info += f"BEM = {baseline_scores['BEM']:.1f}% | "
    if baseline_scores['BEM_NoYaw']: b_info += f"BEM_NoYaw = {baseline_scores['BEM_NoYaw']:.1f}%"
    fig.text(0.5, 0.935, b_info, ha='center', fontsize=11, color='gray', style='italic')
    
    # 1. Convergence
    ax_h = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols)
    ax_h.plot(history['epoch'], history['train_loss'], color='blue', label='Train MSE')
    ax_h.axhline(y=test_loss, color='red', linestyle='--', label=f'Test MSE: {test_loss:.4e}')
    ax_h.set_yscale('log'); ax_h.set_title("Convergence", fontsize=14); ax_h.grid(True, alpha=0.3); ax_h.legend()

    plot_cols = [('Fn', 'Force Normale'), ('Ft', 'Force Tangentielle')]
    if inter == 'v': plot_cols += [(var_map[inter][0], var_map[inter][1]), (var_map[inter][2], var_map[inter][3])]

    axes_th, axes_r = [], []
    for j, (key, label) in enumerate(plot_cols):
        ax_t = plt.subplot2grid((n_rows, n_cols), (1, j))
        ax_t.set_title(f"{label}\nRMSE ML: {metrics[key+'_abs']:.2f} ({metrics[key+'_rel']:.1f}%)", fontsize=11, fontweight='bold')
        ax_t.set_xlabel("Azimut θ [deg]"); ax_t.grid(True, alpha=0.2)
        axes_th.append(ax_t)
        
        ax_rad = plt.subplot2grid((n_rows, n_cols), (2, j))
        ax_rad.set_title(f"Profil spatial: {label}", fontsize=11, fontweight='bold')
        ax_rad.set_xlabel("Rayon r [m]"); ax_rad.grid(True, alpha=0.2)
        axes_r.append(ax_rad)

    for i, r_val in enumerate(sections_r):
        dtr = df_res_tr[df_res_tr['r'] == r_val].sort_values('theta')
        dte = df_res_te[df_res_te['r'] == r_val].sort_values('theta')
        c = colors_r[i]
        for j, (key, _) in enumerate(plot_cols):
            axes_th[j].plot(dtr['theta'], dtr[f'{key}_SVEN'], color=c, marker='o', ls='none', ms=4, alpha=0.3)
            axes_th[j].plot(dte['theta'], dte[f'{key}_SVEN'], color=c, marker='*', ls='none', ms=7, alpha=0.7)
            full_p = pd.concat([dtr, dte]).sort_values('theta')
            axes_th[j].plot(full_p['theta'], full_p[f'{key}_pred'], color=c, ls='-', lw=2.5)

    for i, th_val in enumerate(sections_th):
        dtr = df_res_tr[df_res_tr['theta'] == th_val].sort_values('r')
        dte = df_res_te[df_res_te['theta'] == th_val].sort_values('r')
        c = colors_th[i]
        for j, (key, _) in enumerate(plot_cols):
            axes_r[j].plot(dtr['r'], dtr[f'{key}_SVEN'], color=c, marker='o', ls='none', ms=4, alpha=0.4)
            axes_r[j].plot(dte['r'], dte[f'{key}_SVEN'], color=c, marker='*', ls='none', ms=7, alpha=0.8)
            full_p = pd.concat([dtr, dte]).sort_values('r')
            axes_r[j].plot(full_p['r'], full_p[f'{key}_pred'], color=c, ls='-', lw=2.5)

    rad_h = [Line2D([0], [0], color=colors_r[i], lw=3) for i in range(len(sections_r))]
    axes_th[0].legend(rad_h, [f"r={r:.2f}m ({p*100:.0f}%)" for r, p in zip(sections_r, TARGET_PERCENTAGES)], 
                      loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, title="Rayons (% Span)", frameon=False)
    
    style_h = [Line2D([0], [0], color='gray', marker='o', ls='none'), 
               Line2D([0], [0], color='gray', marker='*', ls='none'), 
               Line2D([0], [0], color='black', ls='-', lw=2)]
    axes_th[1].legend(style_h, ["SVEN Train", "SVEN Test", "Prédiction ML"], loc='upper center', 
                      bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    
    th_h = [Line2D([0], [0], color=colors_th[i], lw=3) for i in range(len(sections_th))]
    axes_r[0].legend(th_h, [f"θ={th:.1f}°" for th in sections_th], loc='upper center', 
                     bbox_to_anchor=(0.5, -0.22), ncol=3, title="Azimuts", frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    safe_rank_label = rank_label.replace(' ','_').replace('é','e')
    plt.savefig(os.path.join(img_dir, f"summary_{safe_rank_label}_{model_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_family_summary(perf_dir="performance/"):
    recap_file = os.path.join(perf_dir, "recap_scores_globaux.csv")
    if not os.path.exists(recap_file): return
        
    df_recap = pd.read_csv(recap_file)
    df_recap['Modele_clean'] = df_recap['Modele'].str.strip()
    df_recap = df_recap.sort_values(by="Score_Global_%", ascending=True)
    
    for f in ['GR', 'GA', 'L']:
        b_bem = df_recap[df_recap['Modele_clean'] == f'Baseline_BEM_{f}']
        b_noyaw = df_recap[df_recap['Modele_clean'] == f'Baseline_BEM_NoYaw_{f}']
        scores_b = {
            'BEM': b_bem['Score_Global_%'].values[0] if not b_bem.empty else None,
            'BEM_NoYaw': b_noyaw['Score_Global_%'].values[0] if not b_noyaw.empty else None
        }
        
        df_fam = df_recap[df_recap['Modele_clean'].str.startswith(f + '_') & ~df_recap['Modele_clean'].str.startswith('Baseline')]
        
        if not df_fam.empty:
            best, worst = df_fam.iloc[0], df_fam.iloc[-1]
            retrain_and_plot(best['Modele_clean'], f"Meilleur {f}", best['Score_Global_%'], scores_b)
            if best['Modele_clean'] != worst['Modele_clean']:
                retrain_and_plot(worst['Modele_clean'], f"Pire {f}", worst['Score_Global_%'], scores_b)

if __name__ == "__main__":
    generate_family_summary()