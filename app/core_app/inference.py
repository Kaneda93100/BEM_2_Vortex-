"""Pipeline d'inférence générique : charge un modèle entraîné (ou le baseline BEM) et produit
les prédictions Fn_pred/Ft_pred (+ V_eff_pred/alpha_pred pour les modèles à intermédiaire 'v')
sur un DataFrame au format train/test (colonnes yaw, TSR, r, theta, + colonnes BEM si besoin).
"""
import json
import pickle

import numpy as np
import torch

from app.core_app.bootstrap import REPO_ROOT
from app.core_app.models_registry import BASELINE_BEM, parse_model_name
from core.config import get_ae_residual_key
from core.models import (
    ConvolutionalAutoencoder,
    LinearAutoencoder,
    TorchScaler,
    TurbineCNN,
    TurbineMLP,
    adapt_ae_output_to_target,
    gv_to_gm_format,
)
from core.physics import compute_V_app
from training.src.data_loader import format_bem_as_Y, format_data, get_D_tensor
from training.src.evaluate import reconstruct_predictions


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_interleaved_flat(preds_coeffs: np.ndarray, is_cnn: bool) -> np.ndarray:
    """(N, 2, H, W) canal-bloc -> (N, 2*H*W) entrelacé [c1,c2,c1,c2,...] (format attendu par
    reconstruct_predictions, identique pour GM et GV)."""
    if not is_cnn:
        return preds_coeffs
    n = preds_coeffs.shape[0]
    flat = np.zeros((n, preds_coeffs.shape[2] * preds_coeffs.shape[3] * 2), dtype=np.float32)
    flat[:, 0::2] = preds_coeffs[:, 0].reshape(n, -1)
    flat[:, 1::2] = preds_coeffs[:, 1].reshape(n, -1)
    return flat


def _predict_baseline_bem(df):
    df_res = df.copy()
    df_res["Fn_pred"] = df_res["Fn_BEM"]
    df_res["Ft_pred"] = df_res["Ft_BEM"]
    if "V_eff_BEM" in df_res.columns:
        df_res["V_eff_pred"] = df_res["V_eff_BEM"]
        df_res["alpha_pred"] = df_res["alpha_BEM"]
    return df_res


def _with_dummy_truth_columns(df):
    """format_data() lit toujours Fn_SVEN/Ft_SVEN/V_eff_SVEN/alpha_SVEN pour construire la
    cible Y (même en inférence pure). Sur un point généré (page Générateur), il n'existe pas de
    vérité terrain : on ajoute des colonnes à 0.0 uniquement pour satisfaire la forme attendue —
    ces valeurs n'influencent en rien X_data ni la prédiction du modèle."""
    missing = [c for c in ("Fn_SVEN", "Ft_SVEN", "V_eff_SVEN", "alpha_SVEN") if c not in df.columns]
    if not missing:
        return df
    df = df.copy()
    for col in missing:
        df[col] = 0.0
    return df


def predict_model(df, model_name: str, device=None):
    """Retourne une copie de `df` enrichie de Fn_pred/Ft_pred (et V_eff_pred/alpha_pred si le
    modèle a un intermédiaire 'v'). `df` doit contenir les colonnes BEM (*_BEM) si le modèle en a
    besoin (résiduelle '1', '2', '2+', ou baseline_bem)."""
    if model_name == BASELINE_BEM:
        return _predict_baseline_bem(df)

    df = _with_dummy_truth_columns(df)
    device = device or _device()
    info = parse_model_name(model_name)
    entree, residuelle, inter = info["entree"], info["residuelle"], info["inter"]
    ae_nature, ae_dim, option = info["ae_nature"], info["ae_dim"], info["option"]
    has_ae = ae_dim > 0
    has_plus = "+" in residuelle
    is_cnn = entree == "GM"

    X_data, Y_data = format_data(df, entree, residuelle, inter, is_train=False, device=device)

    scaler_Y_path = REPO_ROOT / "training" / "scalers" / f"scaler_Y_{entree}_{residuelle}_{inter}.pkl"
    with open(scaler_Y_path, "rb") as f:
        scaler_Y = pickle.load(f)

    ae_model = None
    if has_ae:
        res_base = get_ae_residual_key(residuelle)
        ae_key = f"{res_base}_{inter}_D{ae_nature}{ae_dim}"
        with open(REPO_ROOT / "training" / "hyperparametres" / "ae_hyperparameters.json") as f:
            ae_hps = json.load(f)[ae_key]
        if ae_nature == "M":
            ae_model = ConvolutionalAutoencoder(
                in_channels=2, latent_dim=ae_dim,
                depth=ae_hps["ae_depth"], base_filters=ae_hps["ae_base_filters"], device=device,
            ).to(device)
        else:
            ae_model = LinearAutoencoder(
                in_features=5184, latent_dim=ae_dim,
                n_layers=ae_hps["ae_depth"], device=device,
            ).to(device)
        ae_model.load_state_dict(torch.load(
            REPO_ROOT / "training" / "models" / "ae" / f"ae_{ae_key}.pth", map_location=device,
        ))
        ae_model.eval()

    if has_plus and has_ae:
        n_scalaires = 2 if "TSR" in df.columns else 1
        with torch.no_grad():
            y_bem = format_bem_as_Y(df, entree, inter, scaler_Y, device)
            if entree == "GV":
                y_bem_gm = gv_to_gm_format(y_bem)
                y_bem_in = y_bem_gm.reshape(y_bem.size(0), -1) if ae_nature == "V" else y_bem_gm
                z_bem = ae_model.encode(y_bem_in)
                X_data = torch.cat([X_data[:, :n_scalaires], z_bem], dim=1)
            else:  # GM : z_bem broadcasté en canaux constants (N, ae_dim, 36, 72)
                y_bem_in = y_bem.reshape(y_bem.size(0), -1) if ae_nature == "V" else y_bem
                z_bem = ae_model.encode(y_bem_in)
                zb = z_bem[:, :, None, None].expand(-1, -1, 36, 72).contiguous()
                X_data = torch.cat([X_data[:, :-2], zb], dim=1)

    hp_path = REPO_ROOT / "training" / "hyperparametres" / f"{entree.lower()}_hyperparameters.json"
    hp_key = f"{entree}_{residuelle}_{inter}_{'DXY' if has_ae else 'D0'}_{option}"
    with open(hp_path) as f:
        hps = json.load(f)[hp_key]

    out_dim = ae_dim if has_ae else Y_data.shape[1]
    if entree == "GV":
        model = TurbineMLP(X_data.shape[1], out_dim, hps["n_layers"], hps["n_neurons"], hps["dropout_rate"], device).to(device)
    else:
        model = TurbineCNN(X_data.shape[1], out_dim, has_ae, ae_dim, hps["n_layers"], hps["base_filters"], hps["dropout_rate"], device).to(device)

    model_path = REPO_ROOT / "training" / "models" / entree / f"{model_name}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Impossible de charger les poids de '{model_name}' : l'architecture décrite par "
            f"{hp_key} dans {hp_path.name} (n_layers/n_neurons/base_filters) ne correspond pas "
            f"à celle du fichier {model_path.name}."
        ) from exc
    model.eval()

    scaler_Y_torch = TorchScaler(scaler_Y, device)

    with torch.no_grad():
        preds_raw = model(X_data)
        preds_norm = adapt_ae_output_to_target(ae_model.decode(preds_raw), Y_data) if has_ae else preds_raw
        preds_coeffs = scaler_Y_torch.inverse_transform(preds_norm).cpu().numpy()

    if inter == "f":
        d_tensor = get_D_tensor(df, entree, "cpu").numpy()
        if is_cnn:
            d_exp = np.stack([d_tensor, d_tensor], axis=1)
            preds_coeffs = preds_coeffs * d_exp
        else:
            preds_coeffs = preds_coeffs * d_tensor.reshape(d_tensor.shape[0], -1)

    preds_flat = _to_interleaved_flat(preds_coeffs, is_cnn)
    df_res = reconstruct_predictions(df, preds_flat, entree, residuelle, inter)

    if inter == "v":
        v_app = df_res["v_app"].values if "v_app" in df_res.columns else compute_V_app(df_res)
        if str(residuelle) == "1":
            an_res, at_res = preds_flat[:, 0::2].flatten(), preds_flat[:, 1::2].flatten()
            if "an_BEM" in df_res.columns:
                an_bem, at_bem = df_res["an_BEM"].values, df_res["at_BEM"].values
            else:
                alpha_bem_rad = np.radians(df_res["alpha_BEM"].values)
                v_eff_bem = df_res["V_eff_BEM"].values
                an_bem = np.sin(alpha_bem_rad) * v_eff_bem / v_app
                at_bem = np.cos(alpha_bem_rad) * v_eff_bem / v_app
            an_abs, at_abs = an_res + an_bem, at_res + at_bem
        else:
            an_abs, at_abs = preds_flat[:, 0::2].flatten(), preds_flat[:, 1::2].flatten()

        alpha_rad = np.arctan2(an_abs, at_abs)
        df_res["V_eff_pred"] = v_app * np.sqrt(an_abs ** 2 + at_abs ** 2)
        df_res["alpha_pred"] = np.degrees(alpha_rad)

    return df_res
