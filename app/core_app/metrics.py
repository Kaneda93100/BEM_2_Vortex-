"""Métriques d'erreur (Absolue / Relative A / Relative B) pour les pages Test LHS / Test Uniforme."""
import numpy as np
import pandas as pd

from core.physics import compute_cp, compute_dynamic_pressure_D

ABSOLUE = "Erreur Absolue"
RELATIVE = "Erreur Relative (%)"


def pointwise_error(pred, true, color_mode: str, option: str, is_ft: bool, D=None):
    """Erreur ponctuelle (r,theta) pour la coloration des cartes polaires.

    - Absolue : |pred - vrai|, en N/m (indépendant de A/B).
    - Relative : option A -> |pred-vrai|/|vrai| (F_t plancher à 1.0) ; option B -> |pred-vrai|/D.
    """
    diff = np.abs(pred - true)
    if color_mode == ABSOLUE:
        return diff
    if option == "A":
        denom = np.maximum(np.abs(true), 1.0) if is_ft else np.abs(true)
        return diff / denom * 100.0
    if D is None:
        raise ValueError("D (pression dynamique locale) est requis pour l'option B.")
    return diff / D * 100.0


def local_relative_score(df_slice: pd.DataFrame, option: str) -> float:
    """Score relatif (%) (option A ou B), agrégé (RMSE Fn + RMSE Ft) sur les
    lignes de df_slice uniquement (ex : les 2592 points (r,theta) d'un seul couple (yaw,TSR))."""
    Fn_p, Ft_p = df_slice["Fn_pred"].values, df_slice["Ft_pred"].values
    Fn_s, Ft_s = df_slice["Fn_SVEN"].values, df_slice["Ft_SVEN"].values
    if option == "A":
        err_fn = (Fn_p - Fn_s) / np.abs(Fn_s)
        err_ft = (Ft_p - Ft_s) / np.maximum(np.abs(Ft_s), 1.0)
    else:
        D = compute_dynamic_pressure_D(df_slice)
        err_fn = (Fn_p - Fn_s) / D
        err_ft = (Ft_p - Ft_s) / D
    return float((np.sqrt(np.mean(err_fn ** 2)) + np.sqrt(np.mean(err_ft ** 2))) * 100.0)


def cp_ct_table(df_pred: pd.DataFrame, pred_prefix: str, true_prefix: str = "SVEN") -> pd.DataFrame:
    """Calcule Cp/Ct pour la colonne de prédiction `pred_prefix` (ex: 'pred', 'BEM', 'Castor') et
    la référence `true_prefix` (par défaut 'SVEN'), regroupés par (yaw[, TSR]).

    Retourne un DataFrame avec colonnes yaw[, TSR], Cp_pred, Ct_pred, Cp_true, Ct_true.
    """
    df_p = compute_cp(df_pred, f"Fn_{pred_prefix}", f"Ft_{pred_prefix}")
    df_p = df_p.rename(columns={f"Cp_{pred_prefix}": "Cp_pred", f"Ct_{pred_prefix}": "Ct_pred"})
    df_t = compute_cp(df_pred, f"Fn_{true_prefix}", f"Ft_{true_prefix}")
    df_t = df_t.rename(columns={f"Cp_{true_prefix}": "Cp_true", f"Ct_{true_prefix}": "Ct_true"})

    merge_keys = ["yaw", "TSR"] if "TSR" in df_pred.columns else ["yaw"]
    keep_p = merge_keys + ["Cp_pred", "Ct_pred"]
    keep_t = merge_keys + ["Cp_true", "Ct_true"]
    merged = pd.merge(df_p[keep_p], df_t[keep_t], on=merge_keys)

    for coef in ("Cp", "Ct"):
        merged[f"{coef}_abs_err"] = np.abs(merged[f"{coef}_pred"] - merged[f"{coef}_true"])
        merged[f"{coef}_rel_err"] = merged[f"{coef}_abs_err"] / np.abs(merged[f"{coef}_true"]) * 100.0
    return merged
