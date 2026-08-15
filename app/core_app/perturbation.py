"""Perturbation gaussienne locale de l'entrée BEM (page Corrélation), pour une étude de
sensibilité : on perturbe une composante (Fn/an ou Ft/at) de la BEM au voisinage d'un rayon ou
d'un azimut choisi, et on regarde l'effet sur la prédiction d'un modèle par rapport à la
prédiction non perturbée.

Convention retenue pour le voisinage N (nombre pair de points de grille, ex. 0 à 12 pour r,
0 à 24 pour theta) : demi-largeur de N/2 positions d'INDICE de grille de part et d'autre du
point choisi. Cette demi-largeur sert de largeur à mi-hauteur (HWHM) du poids gaussien
(sigma = (N/2) / sqrt(2*ln 2)) et de frontière du cadre tracé sur les cartes polaires
N=0 dégénère en une perturbation ponctuelle (poids = indicatrice).

Formule d'intensité (% ∈ [50, 500]), "recopie proportionnelle" : au centre (poids=1), la valeur
perturbée vaut exactement intensité% de la valeur originale ; loin du centre (poids→0), la valeur
originale est inchangée :
    perturbée = originale * (1 + poids * (intensité/100 - 1))
"""
import numpy as np
import pandas as pd

_HWHM_TO_SIGMA = 1.0 / np.sqrt(2.0 * np.log(2.0))


def _grid_and_index(df: pd.DataFrame, axis: str):
    """Grille triée des valeurs uniques de `axis` ('r' ou 'theta') dans `df`, et l'indice de
    grille de chaque ligne de `df` sur cette grille."""
    grid = np.sort(df[axis].unique())
    idx_of_value = {v: i for i, v in enumerate(grid)}
    row_idx = df[axis].map(idx_of_value).to_numpy()
    return grid, row_idx


def _nearest_grid_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def gaussian_weights(df: pd.DataFrame, axis: str, center_value: float, neighborhood: int) -> np.ndarray:
    """Poids gaussien (1 au centre, décroît en s'éloignant) pour chaque ligne de `df`, aligné sur
    df.index. `axis` : 'r' (distance linéaire) ou 'theta' (distance circulaire, période = taille
    de la grille). `neighborhood` : voisinage pair en positions d'indice (0 => pic ponctuel)."""
    if axis not in ("r", "theta"):
        raise ValueError(f"axis doit être 'r' ou 'theta', reçu {axis!r}")

    grid, row_idx = _grid_and_index(df, axis)
    n = len(grid)
    i0 = _nearest_grid_index(grid, center_value)

    raw_dist = np.abs(row_idx - i0)
    if axis == "theta":
        dist = np.minimum(raw_dist, n - raw_dist)
    else:
        dist = raw_dist

    if neighborhood <= 0:
        return (dist == 0).astype(float)

    sigma = (neighborhood / 2.0) * _HWHM_TO_SIGMA
    return np.exp(-0.5 * (dist / sigma) ** 2)


def neighborhood_bounds(df: pd.DataFrame, axis: str, center_value: float, neighborhood: int) -> dict:
    """Bornes physiques (valeurs de grille) à N/2 positions d'indice du point choisi, pour tracer
    le cadre du voisinage perturbé. Retourne {'low': ..., 'high': ..., 'wraps': bool} ; `wraps`
    est True si l'intervalle traverse la coupure 360°->0° (theta uniquement)."""
    grid, _ = _grid_and_index(df, axis)
    n = len(grid)
    i0 = _nearest_grid_index(grid, center_value)
    half = neighborhood // 2

    if axis == "theta":
        low_idx = (i0 - half) % n
        high_idx = (i0 + half) % n
        wraps = (i0 - half) < 0 or (i0 + half) >= n
        return {"low": float(grid[low_idx]), "high": float(grid[high_idx]), "wraps": wraps}

    low_idx = max(0, i0 - half)
    high_idx = min(n - 1, i0 + half)
    return {"low": float(grid[low_idx]), "high": float(grid[high_idx]), "wraps": False}


def perturb_bem(
    df: pd.DataFrame, bem_suffix: str, inter: str, component: str,
    axis: str, center_value: float, neighborhood: int, intensity_pct: float,
) -> pd.DataFrame:
    """Copie de `df` avec la composante BEM `component` ('Fn' ou 'Ft') perturbée localement.
    `inter` : 'f' (perturbe directement Fn_BEM_{suffix}/Ft_BEM_{suffix}) ou 'v' (perturbe la
    composante physique équivalente an*V_app/at*V_app, en reconstruisant V_eff_BEM_{suffix}/
    alpha_BEM_{suffix} par la transformation inverse)."""
    if component not in ("Fn", "Ft"):
        raise ValueError(f"component doit être 'Fn' ou 'Ft', reçu {component!r}")

    weight = gaussian_weights(df, axis, center_value, neighborhood)
    factor = 1.0 + weight * (intensity_pct / 100.0 - 1.0)

    df2 = df.copy()
    if inter == "f":
        col = f"Fn_BEM_{bem_suffix}" if component == "Fn" else f"Ft_BEM_{bem_suffix}"
        df2[col] = df2[col].to_numpy() * factor
        return df2

    v_eff_col, alpha_col = f"V_eff_BEM_{bem_suffix}", f"alpha_BEM_{bem_suffix}"
    alpha_rad = np.radians(df2[alpha_col].to_numpy())
    v_eff = df2[v_eff_col].to_numpy()
    an = v_eff * np.sin(alpha_rad)
    at = v_eff * np.cos(alpha_rad)

    if component == "Fn":
        an = an * factor
    else:
        at = at * factor

    df2[v_eff_col] = np.sqrt(an ** 2 + at ** 2)
    df2[alpha_col] = np.degrees(np.arctan2(an, at))
    return df2
