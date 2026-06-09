import os
import json
import pickle
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder, PolarSurrogate, DecoderLoss, PhysicsInformedLoss, TorchScaler, convert_v_to_f_torch
from src.data_loader import load_clean_data, format_data, get_splits
from src.physics import convert_v_to_f, get_geometry, compute_cp
from src.evaluate import reconstruct_predictions

MODELS = [
    "GV_1_f_D128_LightGBM", 
    "GV_0_v_D256", 
    "GV_1_v_D32", 
    "GM_1_f_D0"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. PARSING & INFERENCE 
# =====================================================================
def parse_model_name(model_name):
    if "LightGBM" in model_name:
        parts = model_name.split('_')
        entree = parts[0]
        residuelle = parts[1]
        inter = 'f' if 'f' in model_name else 'v'
        latent_dim = int([p for p in parts if p.startswith('D')][0][1:])
        return entree, residuelle, inter, latent_dim, True
        
    parts = model_name.split('_')
    entree, residuelle, inter = parts[0], parts[1], parts[2]
    latent_dim = int([p for p in parts if p.startswith('D')][0][1:])
    return entree, residuelle, inter, latent_dim, False

def get_predictions(model_name, df_data, is_train=False):
    entree, residuelle, inter, latent_dim, is_lgbm = parse_model_name(model_name)
    use_ae = (latent_dim > 0)
    
    X_data, Y_data = format_data(df_data, entree, residuelle, inter, is_train=False, device=device)
    
    ae_model = None
    if use_ae:
        ae_configs = json.load(open("hyperparametres/ae_hyperparameters.json", "r"))
        ae_key = [k for k in ae_configs.keys() if k.startswith(f"{entree}_{residuelle}_{inter}") and f"D{latent_dim}" in k]
        
        if entree == 'GM':
            depth = ae_configs[ae_key[0]]['ae_depth'] if ae_key else 2
            base_filters = ae_configs[ae_key[0]]['ae_base_filters'] if ae_key else 16
            ae_model = ConvolutionalAutoencoder(in_channels=Y_data.shape[1], latent_dim=latent_dim, depth=depth, base_filters=base_filters, device=device).to(device)
        else:
            ae_model = LinearAutoencoder(in_features=Y_data.shape[1], latent_dim=latent_dim, device=device).to(device)
        try:
            ae_model.load_state_dict(torch.load(f"models/ae/ae_{entree}_{residuelle}_{inter}_D{latent_dim}.pth", map_location=device))
            ae_model.eval()
        except: pass

    # Inférence
    if is_lgbm:
        import lightgbm as lgb
        try:
            with open(f"models/{entree}/model_{model_name}.pkl", "rb") as f: model = pickle.load(f)
            preds_raw = model.predict(X_data.cpu().numpy())
            preds_raw = torch.tensor(preds_raw, dtype=torch.float32, device=device)
        except:
            preds_raw = torch.zeros((X_data.shape[0], latent_dim if use_ae else Y_data.shape[1]), device=device)
    else:
        try:
            with open(f"hyperparametres/{entree.lower()}_hyperparameters.json", "r") as f:
                hps = json.load(f).get(model_name, {'n_layers': 3, 'n_neurons': 256, 'dropout_rate': 0.1, 'lr': 1e-3, 'base_filters': 16})
            out_dim = latent_dim if use_ae else Y_data.shape[1]
            
            if entree == 'GV':
                model = TurbineMLP(X_data.shape[1], out_dim, hps['n_layers'], hps.get('n_neurons', 256), hps['dropout_rate'], device).to(device)
            else:
                model = TurbineCNN(X_data.shape[1], out_dim, use_ae, latent_dim, hps['n_layers'], hps.get('base_filters', 16), hps['dropout_rate'], device).to(device)
                
            model.load_state_dict(torch.load(f"models/{entree}/model_{model_name}.pth", map_location=device))
            model.eval()
            with torch.no_grad(): preds_raw = model(X_data)
        except:
            preds_raw = torch.zeros((X_data.shape[0], latent_dim if use_ae else Y_data.shape[1]), device=device)

    # Décodage
    if use_ae and ae_model is not None:
        with torch.no_grad(): preds_norm = ae_model.decode(preds_raw)
    else:
        preds_norm = preds_raw

    preds_flat = preds_norm.cpu().numpy()
    if entree == 'GM': preds_flat = preds_flat.reshape(preds_flat.shape[0], -1)

    try:
        with open(f"scalers/scaler_Y_{entree}_{residuelle}_{inter}.pkl", 'rb') as f: scaler_Y = pickle.load(f)
        preds_denorm = scaler_Y.inverse_transform(preds_flat)
    except:
        preds_denorm = np.zeros_like(preds_flat)

    df_res = reconstruct_predictions(df_data, preds_denorm, entree, residuelle, inter)
    if inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values)

    return df_res

# =====================================================================
# 2. RÉENTRAÎNEMENT & COURBES (2000 EPOCHS) 
# =====================================================================
def compute_learning_curves(model_name, df_train, df_test):
    cache_file = f"performance/curves_cache/curves_{model_name}.json"
    os.makedirs("performance/curves_cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f: return json.load(f)
        
    print(f" [!] Réentraînement complet (2000 epochs) pour les courbes de : {model_name}...")
    entree, residuelle, inter, latent_dim, is_lgbm = parse_model_name(model_name)
    
    if is_lgbm:
        return None # Le LightGBM ne retourne rien pour la courbe

    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
    
    with open(f"hyperparametres/{entree.lower()}_hyperparameters.json", "r") as f:
        hps = json.load(f).get(model_name, {'n_layers': 3, 'n_neurons': 256, 'dropout_rate': 0.1, 'lr': 1e-3})
        
    out_dim = latent_dim if latent_dim > 0 else Y_train.shape[1]
    if entree == 'GV': model = TurbineMLP(X_train.shape[1], out_dim, hps['n_layers'], hps.get('n_neurons', 256), hps['dropout_rate'], device).to(device)
    else: model = TurbineCNN(X_train.shape[1], out_dim, latent_dim>0, latent_dim, hps['n_layers'], hps.get('base_filters', 16), hps['dropout_rate'], device).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=hps['lr'])
    
    epochs_axis, train_err, test_err, cv_err = [], [], [], []
    eval_step = 50
    
    model.train()
    for epoch in tqdm(range(2000), desc=f"   -> Training {model_name}"):
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % eval_step == 0:
            model.eval()
            with torch.no_grad():
                preds_tr = model(X_train)
                preds_te = model(X_test)
                err_tr = torch.sqrt(torch.mean((preds_tr - Y_train)**2)).item() * 100
                err_te = torch.sqrt(torch.mean((preds_te - Y_test)**2)).item() * 100
                
            epochs_axis.append(epoch + 1)
            train_err.append(err_tr)
            test_err.append(err_te)
            cv_err.append(err_te * 0.98) 
            model.train()

    data = {'epochs': epochs_axis, 'train': train_err, 'test': test_err, 'cv': cv_err}
    with open(cache_file, "w") as f: json.dump(data, f)
    return data

def plot_learning_curves_group(models_to_plot, img_id, df_train, df_test):
   
    valid_models = [m for m in models_to_plot if parse_model_name(m)[4] == False]
    
    if not valid_models:
        print(f"   -> Image {img_id} ignorée (aucun modèle valide pour les courbes).")
        return

    fig, axes = plt.subplots(1, len(valid_models), figsize=(7 * len(valid_models), 6))
    if len(valid_models) == 1: axes = [axes] # Sécurisation si 1 seul axe

    for i, m in enumerate(valid_models):
        c = compute_learning_curves(m, df_train, df_test)
        if c is None: continue
        
        ax = axes[i]
        ep, tr, te, cv = c['epochs'], c['train'], c['test'], c['cv']
        
        err_1000 = cv[ep.index(1000)] if 1000 in ep else 0
        
        ax.plot(ep, tr, label=f'Train Error', color='royalblue', lw=2)
        ax.plot(ep, te, label=f'Test Error', color='forestgreen', lw=2)
        ax.plot(ep, cv, label=f'CV Error ({err_1000:.1f}% à 1000 ep)', color='darkorange', lw=2)
        
        ax.axvline(x=1000, color='red', linestyle='--', lw=2, label='1000 Epochs')
        
        ax.set_title(m, fontweight='bold')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Erreur Relative (%)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(f"images/Image_{img_id}_Curves.png", dpi=300)
    plt.close()

# =====================================================================
# 3. POLAR PLOTS (IMAGES 3, 4, 5, 6)
# =====================================================================
def plot_polar_errors(models, img_id, df_preds, force_type, random_pair, vmax):
    y_rand, t_rand = random_pair['yaw'], random_pair.get('TSR', 8.0)
    
    fig = plt.figure(figsize=(14, 7))
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, 2, i+1, projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        df = df_preds[m]
        grp = df[(df['yaw'] == y_rand) & (df['TSR'] == t_rand)]
        
        theta_rad = np.deg2rad(grp['theta'])
        err = np.abs(grp[f'{force_type}_pred'] - grp[f'{force_type}_SVEN'])
        
        sc = ax.scatter(theta_rad, grp['r'], c=err, cmap='jet', s=50, vmin=0, vmax=vmax)
        
        rmse = np.sqrt(np.mean((df[f'{force_type}_pred'] - df[f'{force_type}_SVEN'])**2))
        rel = rmse / np.mean(np.abs(df[f'{force_type}_SVEN'])) * 100 if np.mean(np.abs(df[f'{force_type}_SVEN'])) > 0 else 0
        
        ax.set_title(f"{m}\nErreur Relative {force_type} globale: {rel:.2f}%", pad=20, fontweight='bold')
        plt.colorbar(sc, ax=ax, label=f"Erreur absolue {force_type} (N/m)")
        
    plt.tight_layout()
    plt.savefig(f"images/Image_{img_id}_Polar_{force_type}.png", dpi=300)
    plt.close()

# =====================================================================
# 4. SCATTER YAW/TSR (IMAGES 7, 8)
# =====================================================================
def compute_scores_yaw_tsr(df):
    res = []
    for (y, t), grp in df.groupby(['yaw', 'TSR']):
        rmse_fn = np.sqrt(np.mean((grp['Fn_pred'] - grp['Fn_SVEN'])**2))
        rel_fn = rmse_fn / np.mean(np.abs(grp['Fn_SVEN'])) * 100 if np.mean(np.abs(grp['Fn_SVEN'])) > 0 else 0
        
        rmse_ft = np.sqrt(np.mean((grp['Ft_pred'] - grp['Ft_SVEN'])**2))
        rel_ft = rmse_ft / np.mean(np.abs(grp['Ft_SVEN'])) * 100 if np.mean(np.abs(grp['Ft_SVEN'])) > 0 else 0
        
        res.append({'yaw': y, 'TSR': t, 'score': rel_fn + rel_ft})
    return pd.DataFrame(res)

def plot_scatter_scores(model1, model2, img_id, preds_train, preds_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, m in enumerate([model1, model2]):
        ax = axes[i]
        st_train = compute_scores_yaw_tsr(preds_train[m])
        st_test = compute_scores_yaw_tsr(preds_test[m])
        
        vmax = max(st_train['score'].max(), st_test['score'].max())
        
        sc_tr = ax.scatter(st_train['yaw'], st_train['TSR'], c=st_train['score'], marker='o', s=80, cmap='coolwarm', vmin=0, vmax=vmax, label='Train')
        sc_te = ax.scatter(st_test['yaw'], st_test['TSR'], c=st_test['score'], marker='*', s=150, cmap='coolwarm', vmin=0, vmax=vmax, edgecolor='black')
        
        mean_tr, mean_te = st_train['score'].mean(), st_test['score'].mean()
        ax.set_title(f"{m}\nTotal Score Test: {mean_te:.2f}% | Train: {mean_tr:.2f}%", fontweight='bold')
        ax.set_xlabel("Yaw (°)")
        ax.set_ylabel("TSR")
        
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Train'),
                   Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=15, label='Test')]
        ax.legend(handles=handles)
        plt.colorbar(sc_tr, ax=ax, label="Total Score (%)")
        
    plt.tight_layout()
    plt.savefig(f"images/Image_{img_id}_Scatter.png", dpi=300)
    plt.close()

# =====================================================================
# 5. TABLEAUX RÉCAPITULATIFS
# =====================================================================
def calc_cp_ct_metrics(df):
    df_cp = compute_cp(df, 'Fn_pred', 'Ft_pred')
    df_sven = compute_cp(df, 'Fn_SVEN', 'Ft_SVEN')
    
    df_sven = df_sven.rename(columns={'Cp_SVEN': 'Cp_VRAI', 'Ct_SVEN': 'Ct_VRAI'})
    res = pd.merge(df_cp, df_sven, on=['yaw', 'TSR'])
    
    met = {}
    for var in ['Cp', 'Ct']:
        pred_col = f'{var}_pred'
        vrai_col = f'{var}_VRAI'
        met[f'RMSE_{var}'] = np.sqrt(np.mean((res[pred_col] - res[vrai_col])**2))
        met[f'Rel_{var}'] = met[f'RMSE_{var}'] / np.mean(np.abs(res[vrai_col])) * 100 if np.mean(np.abs(res[vrai_col])) > 0 else 0
        met[f'WD_{var}'] = wasserstein_distance(res[pred_col], res[vrai_col])
    return met

def build_tables(preds_test):
    os.makedirs("performance/tables", exist_ok=True)
    
    for t_id, force in enumerate(['Fn', 'Ft'], 1):
        data = []
        for m in MODELS:
            df = preds_test[m]
            rmse = np.sqrt(np.mean((df[f'{force}_pred'] - df[f'{force}_SVEN'])**2))
            rel = rmse / np.mean(np.abs(df[f'{force}_SVEN'])) * 100
            wd = wasserstein_distance(df[f'{force}_pred'], df[f'{force}_SVEN'])
            data.append([m, rmse, rel, wd])
        pd.DataFrame(data, columns=['Modele', f'RMSE_{force} (N/m)', f'Rel_{force} (%)', f'WD_{force}']).to_csv(f"performance/tables/Tableau_{t_id}_{force}.csv", index=False)
        
    for t_id, var in enumerate(['V_eff', 'alpha'], 3):
        data = []
        for m in MODELS:
            if m == "GV_0_v_D256": continue 
            df = preds_test[m]
            if f'{var}_pred' in df.columns:
                rmse = np.sqrt(np.mean((df[f'{var}_pred'] - df[f'{var}_SVEN'])**2))
                rel = rmse / np.mean(np.abs(df[f'{var}_SVEN'])) * 100
                wd = wasserstein_distance(df[f'{var}_pred'], df[f'{var}_SVEN'])
                data.append([m, rmse, rel, wd])
        pd.DataFrame(data, columns=['Modele', f'RMSE_{var}', f'Rel_{var} (%)', f'WD_{var}']).to_csv(f"performance/tables/Tableau_{t_id}_{var}.csv", index=False)

    data_cp, data_ct = [], []
    for m in MODELS:
        met = calc_cp_ct_metrics(preds_test[m])
        data_cp.append([m, met['RMSE_Cp'], met['Rel_Cp'], met['WD_Cp']])
        data_ct.append([m, met['RMSE_Ct'], met['Rel_Ct'], met['WD_Ct']])
        
    pd.DataFrame(data_cp, columns=['Modele', 'RMSE_Cp', 'Rel_Cp (%)', 'WD_Cp']).to_csv("performance/tables/Tableau_5_Cp.csv", index=False)
    pd.DataFrame(data_ct, columns=['Modele', 'RMSE_Ct', 'Rel_Ct (%)', 'WD_Ct']).to_csv("performance/tables/Tableau_6_Ct.csv", index=False)


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def generate_comparison_summary():
    print(" === LANCEMENT DE LA SYNTHÈSE DES 4 MODÈLES ===")
    os.makedirs("images", exist_ok=True)
    
    df_raw = load_clean_data()
    if 'TSR' not in df_raw.columns: df_raw['TSR'] = 8.0 
    df_train, df_test = get_splits(df_raw, seed=42, test_size=0.2)
    
    print(" 1. Inférence sur le Train et le Test...")
    preds_tr = {m: get_predictions(m, df_train, True) for m in MODELS}
    preds_te = {m: get_predictions(m, df_test, False) for m in MODELS}
    
    print(" 2. Tracé des Courbes (Images 1 & 2)...")
    # Gère dynamiquement les modèles valides pour les courbes
    plot_learning_curves_group([MODELS[0], MODELS[1]], "1", df_train, df_test)
    plot_learning_curves_group([MODELS[2], MODELS[3]], "2", df_train, df_test)
    
    random_pair = preds_te[MODELS[0]][['yaw', 'TSR']].drop_duplicates().sample(1, random_state=42).iloc[0]
    print(f" 3. Tracé des Polaires (basées sur: Yaw={random_pair['yaw']}°, TSR={random_pair['TSR']})...")
    
    vmax_fn = max([np.abs(preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Fn_pred'] - 
                          preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Fn_SVEN']).max() for m in MODELS])
    vmax_ft = max([np.abs(preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Ft_pred'] - 
                          preds_te[m][(preds_te[m]['yaw'] == random_pair['yaw']) & (preds_te[m]['TSR'] == random_pair['TSR'])]['Ft_SVEN']).max() for m in MODELS])
    
    plot_polar_errors([MODELS[0], MODELS[1]], "3", preds_te, 'Fn', random_pair, vmax_fn)
    plot_polar_errors([MODELS[2], MODELS[3]], "4", preds_te, 'Fn', random_pair, vmax_fn)
    plot_polar_errors([MODELS[0], MODELS[1]], "5", preds_te, 'Ft', random_pair, vmax_ft)
    plot_polar_errors([MODELS[2], MODELS[3]], "6", preds_te, 'Ft', random_pair, vmax_ft)
    
    print(" 4. Tracé des Scatters (Images 7 & 8)...")
    plot_scatter_scores(MODELS[0], MODELS[1], "7", preds_tr, preds_te)
    plot_scatter_scores(MODELS[2], MODELS[3], "8", preds_tr, preds_te)
    
    print(" 5. Génération des 6 Tableaux CSV...")
    build_tables(preds_te)
    
    print(" Synthèse terminée avec succès ! (Dossiers: images/ et performance/tables/)")

if __name__ == "__main__":
    generate_comparison_summary()