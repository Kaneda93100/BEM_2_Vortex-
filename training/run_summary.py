import os
import re
import json
import pickle
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm

# Désactivation des warnings de version Scikit-Learn au chargement des scalers picklés
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from core.models import TurbineMLP, LinearAutoencoder, TurbineLoss, TorchScaler, gv_to_gm_format, adapt_ae_output_to_target
from core.physics import get_geometry
from core.config import (RANDOM_SEED, CV_SPLITS, AE_JSON_PATH, AE_WEIGHTS_DIR,
                          format_scaler_name, format_model_name, format_ae_key)
from training.src.data_loader import load_clean_data, get_splits, subsample_train, format_data, get_D_tensor, format_bem_as_Y
from training.src.evaluate import get_u_inf_tensor

# =====================================================================
# CONFIGURATION
# =====================================================================
MODELS = ["GV_0_f_D0_A_P50", "GV_2+_f_IFP_DV64_A_P50"]

EPOCHS_MAX = 1000
EPOCH_STEP = 10
CHECKPOINTS = list(range(EPOCH_STEP, EPOCHS_MAX + 1, EPOCH_STEP))
VLINE_EPOCH = 500

IMAGES_DIR = "training/performance/images"
CACHE_DIR = "training/performance/curves_cache"

# Palette catégorielle (slots 1/2/3), validée CVD-safe
COLOR_TRAIN = "#2a78d6"
COLOR_TEST = "#eb6834"
COLOR_CV = "#1baf7a"
COLOR_VLINE = "#898781"
COLOR_GRID = "#e1e0d9"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME_RE = re.compile(
    r"^(GM|GV)_(0|1|2\+?)_([fv])(?:_(DUM|IFP|P&P))?_D([A-Z]*)(\d+)_([AB])_P(\d{1,3})$"
)

def parse_model_name(model_name):
    m = MODEL_NAME_RE.match(model_name)
    if not m:
        raise ValueError(f"Nom de modèle non reconnu : '{model_name}'")
    entree, residuelle, inter, bem_suffix, ae_nature, ae_dim, option, pct = m.groups()
    return {
        "entree": entree, "residuelle": residuelle, "inter": inter,
        "bem_suffix": bem_suffix, "ae_nature": ae_nature or None,
        "ae_dim": int(ae_dim), "option": option, "pct": int(pct),
    }

# =====================================================================
# 1. PRÉPARATION DES DONNÉES / MODÈLE (reproduit exactement le pipeline de evaluator())
# =====================================================================
def prepare_model(model_name, df_train_full, df_test):
    info = parse_model_name(model_name)
    entree, residuelle, inter = info["entree"], info["residuelle"], info["inter"]
    bem_suffix, ae_nature, ae_dim = info["bem_suffix"], info["ae_nature"], info["ae_dim"]
    option, pct = info["option"], info["pct"]
    has_ae = ae_dim > 0
    has_plus = '+' in residuelle

    df_train_sub = subsample_train(df_train_full, pct)

    ae_label = "DXY" if has_ae else "D0"
    hp_key = format_model_name(entree, residuelle, inter, ae_label, option, pct, bem_suffix)
    with open(f"training/hyperparametres/{entree.lower()}_hyperparameters.json", "r") as f:
        best_params = json.load(f)[hp_key]

    X_train, Y_train = format_data(df_train_sub, entree, residuelle, inter, is_train=True, device=device, bem_suffix=bem_suffix)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device, bem_suffix=bem_suffix)

    with open(f"training/scalers/scaler_Y_{format_scaler_name(entree, residuelle, inter, bem_suffix)}.pkl", 'rb') as f:
        scaler_Y = pickle.load(f)
    scaler_Y_torch = TorchScaler(scaler_Y, device)

    geom = get_geometry()
    group = df_train_sub[df_train_sub['yaw'] == df_train_sub['yaw'].iloc[0]]
    if 'TSR' in group.columns:
        group = group[group['TSR'] == group['TSR'].iloc[0]]
    group = group.sort_values(['theta', 'r'])
    r_tensor = torch.tensor(group['r'].values, dtype=torch.float32, device=device)
    c_tensor = torch.tensor(np.array([geom.get_chord(r) for r in group['r'].values]), dtype=torch.float32, device=device)

    current_ae = None
    if has_ae:
        ae_key = format_ae_key(residuelle, inter, ae_nature, ae_dim, bem_suffix)
        with open(AE_JSON_PATH, "r") as f:
            ae_config = json.load(f)[ae_key]
        current_ae = LinearAutoencoder(in_features=Y_train.shape[1], latent_dim=ae_dim, n_layers=ae_config['ae_depth'], device=device).to(device)
        current_ae.load_state_dict(torch.load(os.path.join(AE_WEIGHTS_DIR, f"ae_{ae_key}.pth"), map_location=device))
        current_ae.eval()
        for p in current_ae.parameters():
            p.requires_grad_(False)

    if has_plus and current_ae is not None:
        n_scalaires = 2 if 'TSR' in df_train_sub.columns else 1
        with torch.no_grad():
            Y_bem_train = format_bem_as_Y(df_train_sub, entree, inter, scaler_Y, bem_suffix, device)
            Y_bem_test = format_bem_as_Y(df_test, entree, inter, scaler_Y, bem_suffix, device)
            tr_cnn = gv_to_gm_format(Y_bem_train)
            te_cnn = gv_to_gm_format(Y_bem_test)
            y_bem_tr = tr_cnn.reshape(Y_bem_train.size(0), -1) if ae_nature == 'V' else tr_cnn
            y_bem_te = te_cnn.reshape(Y_bem_test.size(0), -1) if ae_nature == 'V' else te_cnn
            z_bem_train = current_ae.encode(y_bem_tr)
            z_bem_test = current_ae.encode(y_bem_te)
            X_train = torch.cat([X_train[:, :n_scalaires], z_bem_train], dim=1)
            X_test = torch.cat([X_test[:, :n_scalaires], z_bem_test], dim=1)

    D_train = get_D_tensor(df_train_sub, entree, device)
    D_test = get_D_tensor(df_test, entree, device)
    u_inf_train = get_u_inf_tensor(df_train_sub, device)

    target_dim = ae_dim if has_ae else Y_train.shape[1]
    model_kwargs = dict(input_dim=X_train.shape[1], output_dim=target_dim, n_layers=best_params['n_layers'],
                         n_neurons=best_params['n_neurons'], dropout_rate=best_params['dropout_rate'], device=device)

    def make_criterion():
        return TurbineLoss(inter=inter, loss_type=option, l1=best_params['l1'], l2=best_params['l2'], l3=best_params['l3'],
                            ae_model=current_ae, scaler_Y=scaler_Y, r_tensor=r_tensor, c_tensor=c_tensor,
                            polar_surrogate=None, device=device)

    return {
        "has_ae": has_ae, "lr": best_params['lr'], "model_kwargs": model_kwargs, "make_criterion": make_criterion,
        "current_ae": current_ae, "scaler_Y_torch": scaler_Y_torch,
        "X_train": X_train, "Y_train": Y_train, "D_train": D_train, "u_inf_train": u_inf_train,
        "X_test": X_test, "Y_test": Y_test, "D_test": D_test,
    }

# =====================================================================
# 2. SCORE A (erreur relative locale Fn/Ft, cf. evaluate.py::compute_phys_score)
# =====================================================================
def compute_score_a(model, ae_model, has_ae, X_eval, Y_eval, D_eval, scaler_Y_torch):
    model.eval()
    with torch.no_grad():
        preds_raw = model(X_eval)
        preds_norm = adapt_ae_output_to_target(ae_model.decode(preds_raw), Y_eval) if has_ae else preds_raw
        coeffs_pred = scaler_Y_torch.inverse_transform(preds_norm)
        coeffs_true = scaler_Y_torch.inverse_transform(Y_eval)

        Fn_p, Ft_p = coeffs_pred[:, 0::2] * D_eval[:, 0::2], coeffs_pred[:, 1::2] * D_eval[:, 1::2]
        Fn_t, Ft_t = coeffs_true[:, 0::2] * D_eval[:, 0::2], coeffs_true[:, 1::2] * D_eval[:, 1::2]

        err_fn = (Fn_p - Fn_t) / torch.abs(Fn_t)
        err_ft = (Ft_p - Ft_t) / torch.clamp(torch.abs(Ft_t), min=1.0)
        score = ((torch.sqrt(torch.mean(err_fn**2)) + torch.sqrt(torch.mean(err_ft**2))) * 100).item()
    model.train()
    return score

# =====================================================================
# 3. ENTRAÎNEMENT AVEC SUIVI DU SCORE A PAR EPOCH
# =====================================================================
def train_curve_main(bundle, epochs, checkpoints, model_name):
    model = TurbineMLP(**bundle["model_kwargs"]).to(device)
    criterion = bundle["make_criterion"]()
    optimizer = torch.optim.Adam(model.parameters(), lr=bundle["lr"])
    checkpoints_set = set(checkpoints)

    train_scores, test_scores = [], []
    for epoch in tqdm(range(1, epochs + 1), desc=f"   -> {model_name} [Train complet]"):
        model.train()
        optimizer.zero_grad()
        preds = model(bundle["X_train"])
        loss = criterion(preds, bundle["Y_train"], D_phys=bundle["D_train"], u_inf=bundle["u_inf_train"])
        loss.backward()
        optimizer.step()

        if epoch in checkpoints_set:
            train_scores.append(compute_score_a(model, bundle["current_ae"], bundle["has_ae"],
                                                 bundle["X_train"], bundle["Y_train"], bundle["D_train"], bundle["scaler_Y_torch"]))
            test_scores.append(compute_score_a(model, bundle["current_ae"], bundle["has_ae"],
                                                bundle["X_test"], bundle["Y_test"], bundle["D_test"], bundle["scaler_Y_torch"]))
    return train_scores, test_scores

def train_curve_cv(bundle, epochs, checkpoints, model_name, n_splits=CV_SPLITS, seed=RANDOM_SEED):
    X_full, Y_full, D_full, u_inf_full = bundle["X_train"], bundle["Y_train"], bundle["D_train"], bundle["u_inf_train"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    checkpoints_set = set(checkpoints)

    fold_curves = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full.cpu().numpy())):
        X_tr, Y_tr, D_tr, u_tr = X_full[train_idx], Y_full[train_idx], D_full[train_idx], u_inf_full[train_idx]
        X_val, Y_val, D_val = X_full[val_idx], Y_full[val_idx], D_full[val_idx]

        model = TurbineMLP(**bundle["model_kwargs"]).to(device)
        criterion = bundle["make_criterion"]()
        optimizer = torch.optim.Adam(model.parameters(), lr=bundle["lr"])

        fold_scores = []
        for epoch in tqdm(range(1, epochs + 1), desc=f"   -> {model_name} [CV fold {fold + 1}/{n_splits}]", leave=False):
            model.train()
            optimizer.zero_grad()
            preds = model(X_tr)
            loss = criterion(preds, Y_tr, D_phys=D_tr, u_inf=u_tr)
            loss.backward()
            optimizer.step()

            if epoch in checkpoints_set:
                fold_scores.append(compute_score_a(model, bundle["current_ae"], bundle["has_ae"],
                                                     X_val, Y_val, D_val, bundle["scaler_Y_torch"]))
        fold_curves.append(fold_scores)

    return np.mean(fold_curves, axis=0).tolist()

def compute_curves(model_name, df_train_full, df_test):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/scoreA_{model_name.replace('+', 'p')}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    print(f" [!] Calcul des courbes Score A ({EPOCHS_MAX} epochs, {CV_SPLITS}-fold CV) : {model_name}...")
    bundle = prepare_model(model_name, df_train_full, df_test)

    train_scores, test_scores = train_curve_main(bundle, EPOCHS_MAX, CHECKPOINTS, model_name)
    cv_scores = train_curve_cv(bundle, EPOCHS_MAX, CHECKPOINTS, model_name)

    data = {"epochs": CHECKPOINTS, "train": train_scores, "test": test_scores, "cv": cv_scores}
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data

# =====================================================================
# 4. TRACÉ (échelle linéaire + échelle log sur le score)
# =====================================================================
def plot_score_a_curves(model_name, curve, out_path):
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, yscale, title in [(ax_lin, 'linear', 'Échelle linéaire'), (ax_log, 'log', 'Échelle logarithmique')]:
        ax.plot(curve['epochs'], curve['train'], color=COLOR_TRAIN, lw=2, label='Train (utilisé)')
        ax.plot(curve['epochs'], curve['test'], color=COLOR_TEST, lw=2, label='Test')
        ax.plot(curve['epochs'], curve['cv'], color=COLOR_CV, lw=2, label='Cross-validation')
        ax.axvline(x=VLINE_EPOCH, color=COLOR_VLINE, linestyle='--', lw=1.5, label=f'{VLINE_EPOCH} epochs')
        ax.set_yscale(yscale)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Score A (%)")
        ax.set_title(title, fontweight='bold')
        ax.grid(True, color=COLOR_GRID, linestyle='-', linewidth=0.8, alpha=0.8)
        ax.legend()

    fig.suptitle(f"Score A vs Epochs — {model_name}", fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def generate_score_a_curves():
    print(" === COURBES SCORE A (Train utilisé / Test / Cross-validation) vs EPOCHS ===")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df_raw = load_clean_data()
    if 'TSR' not in df_raw.columns:
        df_raw['TSR'] = 8.0
    df_train, df_test = get_splits(df_raw, seed=RANDOM_SEED, test_size=0.2)

    for model_name in MODELS:
        curve = compute_curves(model_name, df_train, df_test)
        out_path = f"{IMAGES_DIR}/Courbe_ScoreA_{model_name.replace('+', 'p')}.png"
        plot_score_a_curves(model_name, curve, out_path)
        print(f" [OK] Graphique sauvegardé : {out_path}")

    print(" Terminé.")

if __name__ == "__main__":
    generate_score_a_curves()
