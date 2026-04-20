import os
import glob
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
    """
    Ré-entraîne le modèle en mémoire pour générer la planche graphique.
    Applique la dénormalisation avant le tracé.
    SVEN = Points | ML = Lignes
    """
    print(f"   ▶ Analyse visuelle détaillée : {model_name} ({rank_label})")
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        entree, residuelle, inter = model_name.split('_')
        with open(f"hyperparametres/{model_name}.json", "r") as f:
            hparams = json.load(f)
    except Exception as e:
        print(f"   ⚠️ Erreur : {e}")
        return

    df_train = pd.read_csv(f"data/processed/train_{entree}.csv")
    df_test = pd.read_csv(f"data/processed/test_{entree}.csv")
    
    # NORMALISATION
    X_full_train, Y_full_train = format_data(df_train, entree, residuelle, inter, is_train=True)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False)
    
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_full_train.numpy(), Y_full_train.numpy(), test_size=0.2, random_state=42)
    X_tr, Y_tr = torch.tensor(X_tr), torch.tensor(Y_tr)
    X_val, Y_val = torch.tensor(X_val), torch.tensor(Y_val)
    
    model = TurbineMLP(X_tr.shape[1], Y_tr.shape[1], hparams['n_layers'], hparams['n_neurons'], hparams['dropout_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    patience, best_val_loss, best_weights = 50, float('inf'), None
    
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
        preds_tr, preds_te = model(X_full_train).numpy(), model(X_test).numpy()
        
    # --- DÉNORMALISATION DES PRÉDICTIONS ---
    with open(f"scalers/scaler_Y_{model_name}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
        
    preds_tr = scaler_Y.inverse_transform(preds_tr)
    preds_te = scaler_Y.inverse_transform(preds_te)
    # ---------------------------------------

    df_res_tr = reconstruct_predictions(df_train, preds_tr, entree, residuelle, inter)
    df_res_te = reconstruct_predictions(df_test, preds_te, entree, residuelle, inter)
    
    for df in [df_res_tr, df_res_te]:
        if inter == 'u':
            df['V_eff_pred'], df['alpha_pred'] = convert_u_to_v(df['a_pred'].values, df['phi_pred'].values, df['r'].values)
            df['Fn_pred'], df['Ft_pred'] = convert_v_to_f(df['V_eff_pred'].values, df['alpha_pred'].values, df['r'].values)
        elif inter == 'v':
            df['Fn_pred'], df['Ft_pred'] = convert_v_to_f(df['V_eff_pred'].values, df['alpha_pred'].values, df['r'].values)

    # Calcul des métriques pour tous les graphiques (sur le Test Set)
    metrics = {}
    target_cols = ['Fn', 'Ft']
    if inter == 'v': target_cols += ['V_eff', 'alpha']
    if inter == 'u': target_cols += ['a', 'phi']

    for col in target_cols:
        s, p = df_res_te[f'{col}_SVEN'].values, df_res_te[f'{col}_pred'].values
        rmse = np.sqrt(np.mean((p - s)**2))
        avg = np.mean(np.abs(s))
        rel = (rmse / avg * 100) if avg != 0 else 0
        metrics[f'{col}_abs'], metrics[f'{col}_rel'] = rmse, rel

    _draw_plot(model_name, rank_label, history, df_res_tr, df_res_te, inter, img_dir, metrics)

def _draw_plot(model_name, rank_label, history, df_res_tr, df_res_te, inter, img_dir, metrics):
    n_cols = 4 if inter in ['u', 'v'] else 2
    var_map = {
        'v': ('V_eff', '$V_{eff}$ [m/s]', 'alpha', '$\\alpha$ [deg]'),
        'u': ('a', 'Induction $a$', 'phi', 'Angle flux $\\phi$ [deg]')
    }
    
    rayons = np.sort(df_res_te['r'].unique())
    sections = rayons[np.linspace(0, len(rayons)-1, 6, dtype=int)] if len(rayons) > 6 else rayons
    
    # Couleurs plus contrastées (Palette qualitative tab10)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))

    fig = plt.figure(figsize=(5 * n_cols, 10))
    fig.suptitle(f"Synthèse Modèle : {model_name} ({rank_label})", fontsize=20, fontweight='bold', y=0.96)
    
    # Historique de convergence
    ax_h = plt.subplot2grid((2, n_cols), (0, 0), colspan=n_cols)
    ax_h.plot(history['epoch'], history['train_loss'], color='blue', label='Train MSE')
    ax_h.plot(history['epoch'], history['val_loss'], color='red', label='Val MSE')
    ax_h.set_yscale('log'); ax_h.set_title("Convergence"); ax_h.grid(True, alpha=0.3); ax_h.legend()

    # Définition des colonnes à tracer
    plot_cols = [('Fn', 'Force Normale'), ('Ft', 'Force Tangentielle')]
    if inter in ['v', 'u']:
        plot_cols += [(var_map[inter][0], var_map[inter][1]), (var_map[inter][2], var_map[inter][3])]

    axes = []
    for j, (key, label) in enumerate(plot_cols):
        ax = plt.subplot2grid((2, n_cols), (1, j))
        unit = " [N/m]" if j < 2 else ""
        ax.set_title(f"{label}\nRMSE Test: {metrics[key+'_abs']:.2f}{unit} ({metrics[key+'_rel']:.1f}%)", fontsize=10, fontweight='bold')
        ax.set_xlabel("Azimut [deg]"); ax.grid(True, alpha=0.2)
        axes.append(ax)

    # Tracé superposé
    for i, r_val in enumerate(sections):
        dtr, dte = df_res_tr[df_res_tr['r'] == r_val], df_res_te[df_res_te['r'] == r_val]
        c = colors[i]
        
        for j, (key, _) in enumerate(plot_cols):
            # SVEN (Vérité) = Points (Ronds Train / Étoiles Test)
            axes[j].plot(dtr['theta'], dtr[f'{key}_SVEN'], color=c, marker='o', ls='none', ms=4, alpha=0.3)
            axes[j].plot(dte['theta'], dte[f'{key}_SVEN'], color=c, marker='*', ls='none', ms=7, alpha=0.7)
            # ML (Prédiction) = Trait plein continu
            full_p = pd.concat([dtr, dte]).sort_values('theta')
            axes[j].plot(full_p['theta'], full_p[f'{key}_pred'], color=c, ls='-', lw=2, alpha=0.9)

    # Légendes
    rad_lines = [Line2D([0], [0], color=colors[i], lw=3) for i in range(len(sections))]
    axes[0].legend(rad_lines, [f"r={r:.2f}m" for r in sections], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, title="Sections (Rayons)")
    
    style_el = [Line2D([0], [0], color='gray', marker='o', ls='none'), 
                Line2D([0], [0], color='gray', marker='*', ls='none'), 
                Line2D([0], [0], color='gray', ls='-', lw=2)]
    axes[1].legend(style_el, ["SVEN Train", "SVEN Test", "Prédiction ML"], loc='best', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    safe_label = rank_label.replace(' ', '_')
    plt.savefig(os.path.join(img_dir, f"summary_{safe_label}_{model_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_table(perf_dir="performance/", N_tab=2, N_fig=1):
    """
    Génère le tableau récapitulatif et les planches graphiques.
    N_tab : Nombre de modèles par catégorie (Top, Median, Pire) dans le tableau.
    N_fig : Nombre de modèles par catégorie à tracer en graphique.
    """
    files = glob.glob(os.path.join(perf_dir, "results_*.csv"))
    if not files: return
    
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_all['Score_Global_%'] = df_all['RMSE_Fn_Rel_%'] + df_all['RMSE_Ft_Rel_%']
    df_all = df_all.sort_values(by='Score_Global_%', ascending=True).reset_index(drop=True)
    n = len(df_all)
    
    # Construction dynamique des index pour le Tableau
    if n <= 3 * N_tab:
        idx_list = list(range(n))
        labels = [f"Rang {i+1}" for i in range(n)]
    else:
        top_idx = list(range(N_tab))
        mid_start = (n - N_tab) // 2
        mid_idx = list(range(mid_start, mid_start + N_tab))
        bot_idx = list(range(n - N_tab, n))
        
        idx_list = top_idx + mid_idx + bot_idx
        
        labels = []
        for i in range(N_tab): labels.append(f"TOP {i+1}")
        for i in range(N_tab): labels.append(f"MEDIAN {i+1}")
        for i in range(N_tab, 0, -1): labels.append(f"PIRE {i}") # Ex: PIRE 2, PIRE 1

    df_display = df_all.iloc[idx_list].copy()
    df_display.insert(0, 'Rang', labels)
    
    print("\n" + "🏆 CLASSEMENT FINAL DES MODÈLES (Basé sur l'Erreur Relative Cumulée)")
    print("="*105)
    print(df_display[['Rang', 'Modele', 'Epochs_Conv', 'Score_Global_%', 'RMSE_Fn_Rel_%', 'RMSE_Ft_Rel_%', 'Wasserstein_Fn', 'Wasserstein_Ft']].round(2).to_string(index=False))
    print("="*105)
    print(f"Bilan : {n} modèles évalués. Le meilleur modèle est {df_all.iloc[0]['Modele']}.")
    print("="*105)

    # Construction dynamique des cibles pour les Images (Figures)
    print(f"\n🎨 Génération des planches visuelles ({N_fig} par groupe)...")
    targets = {}
    
    # Top N_fig
    for i in range(min(N_fig, n)):
        targets[i] = f"TOP_{i+1}"
        
    # Median N_fig
    mid_start = (n - N_fig) // 2
    for i in range(min(N_fig, n)):
        idx = mid_start + i
        if idx not in targets and idx < n:
            targets[idx] = f"MEDIAN_{i+1}"
            
    # Bottom N_fig
    for i in range(min(N_fig, n)):
        idx = n - N_fig + i
        if idx not in targets and idx < n:
            targets[idx] = f"PIRE_{N_fig - i}"

    for idx, lbl in targets.items():
        retrain_and_plot(df_all.iloc[idx]['Modele'], lbl)


generate_summary_table()