"""Fonctions de tracé (matplotlib) réutilisées par les pages Test LHS / Test Uniforme."""
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from app.core_app.metrics import ABSOLUE, local_relative_score, pointwise_error
from core.physics import compute_dynamic_pressure_D


def _norm(vmax, log_scale: bool):
    """Normalisation de la colorbar : linéaire [0, vmax] ou logarithmique (plancher positif, les
    erreurs exactement nulles ou négatives n'ont pas de sens sur une échelle log)."""
    if not log_scale:
        return Normalize(vmin=0, vmax=vmax)
    vmin = max(vmax * 1e-3, 1e-9)
    return LogNorm(vmin=vmin, vmax=vmax)


def _clip_for_log(values, log_scale: bool, vmax):
    if not log_scale:
        return values
    vmin = max(vmax * 1e-3, 1e-9)
    return np.clip(values, vmin, None)


SCALE_COMMUNE = "Commune"
SCALE_PROPRE = "Propre"


def _vmax_lookup(errors: dict, shared_scale: bool):
    """Retourne une fonction name -> vmax : un seul vmax global si shared_scale (échelle
    "Commune"), sinon un vmax par clé (échelle "Propre", chaque sous-graphique s'auto-cadre)."""
    if shared_scale:
        vmax = max((err.max() for err in errors.values()), default=1.0)
        vmax = vmax if vmax > 0 else 1.0
        return lambda name: vmax
    return lambda name: (errors[name].max() if errors[name].size and errors[name].max() > 0 else 1.0)


def polar_error_figure(slices: dict, force_col: str, color_mode: str, option: str,
                        log_scale: bool = False, shared_scale: bool = True):
    """slices : {nom_modele: df_filtré_sur_(yaw,TSR)} avec colonnes r, theta,
    {force_col}_pred, {force_col}_SVEN. Renvoie une figure avec une polaire par modèle.
    `shared_scale` : True = échelle de couleur commune à tous les sous-graphiques, False =
    chaque sous-graphique a sa propre échelle (son propre max)."""
    is_ft = force_col == "Ft"
    errors = {}
    for name, df in slices.items():
        D = compute_dynamic_pressure_D(df) if color_mode != ABSOLUE and option == "B" else None
        errors[name] = pointwise_error(
            df[f"{force_col}_pred"].values, df[f"{force_col}_SVEN"].values, color_mode, option, is_ft, D,
        )
    vmax_of = _vmax_lookup(errors, shared_scale)

    fig = Figure(figsize=(7 * len(slices), 6))
    for i, (name, df) in enumerate(slices.items()):
        vmax = vmax_of(name)
        norm = _norm(vmax, log_scale)
        ax = fig.add_subplot(1, len(slices), i + 1, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        theta_rad = np.deg2rad(df["theta"].values)
        err_vals = _clip_for_log(errors[name], log_scale, vmax)
        sc = ax.scatter(theta_rad, df["r"].values, c=err_vals, cmap="jet", s=40, norm=norm)
        score = local_relative_score(df, option)
        unit = "%" if color_mode != ABSOLUE else "N/m"
        cbar_label = f"{color_mode} {force_col} ({unit})"
        ax.set_title(f"{name}\nScore {option} local ({force_col.lower()}, r-theta) : {score:.2f}%", pad=20, fontweight="bold")
        fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    return fig


def _draw_neighborhood_outline(ax, axis: str, bounds: dict, r_min: float, r_max: float):
    """Cadre noir délimitant la zone perturbée sur une carte polaire (ax en projection polar).
    axis='r' : deux cercles (rayons bounds['low']/['high']) sur tout l'azimut.
    axis='theta' : deux rayons (azimuts bounds['low']/['high']), avec repliement 360°->0° géré en
    dessinant le secteur en deux morceaux si bounds['wraps'] est vrai."""
    style = dict(color="black", linewidth=2.5, zorder=5)
    if axis == "r":
        theta_full = np.linspace(0, 2 * np.pi, 200)
        for radius in (bounds["low"], bounds["high"]):
            ax.plot(theta_full, np.full_like(theta_full, radius), **style)
        return

    low_rad, high_rad = np.deg2rad(bounds["low"]), np.deg2rad(bounds["high"])
    if not bounds["wraps"]:
        theta_arc = np.linspace(low_rad, high_rad, 100)
    else:
        theta_arc = np.linspace(low_rad, high_rad + 2 * np.pi, 100) % (2 * np.pi)
    ax.plot([low_rad, low_rad], [r_min, r_max], **style)
    ax.plot([high_rad, high_rad], [r_min, r_max], **style)
    ax.plot(theta_arc, np.full_like(theta_arc, r_max), **style)
    ax.plot(theta_arc, np.full_like(theta_arc, r_min), **style)


def _draw_center_line(ax, axis: str, center_value: float, r_min: float, r_max: float):
    """Pointillé noir marquant le r ou le theta exact qui a été le centre de la perturbation
    (distinct du cadre continu de neighborhood_bounds qui délimite la zone affectée)."""
    style = dict(color="black", linewidth=1.8, linestyle="--", zorder=6)
    if axis == "r":
        theta_full = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta_full, np.full_like(theta_full, center_value), **style)
    else:
        rad = np.deg2rad(center_value)
        ax.plot([rad, rad], [r_min, r_max], **style)


def polar_perturbation_figure(slices: dict, force_col: str, color_mode: str, option: str,
                               axis: str, bounds: dict, center_value: float,
                               log_scale: bool = False, shared_scale: bool = True):
    """slices : {nom_modele: (df_perturbé, df_baseline)}, chacun avec colonnes r, theta,
    {force_col}_pred. Trace l'écart (perturbé vs non perturbé) sur des cartes polaires, avec un
    cadre noir continu délimitant la zone perturbée et un pointillé noir
    marquant le r/theta exact choisi comme centre de la perturbation.Si `shared_scale =True alors
    l'échelle de couleur commune à tous les sous-graphiques sinon elle est propre à chacun."""
    is_ft = force_col == "Ft"
    errors, r_ranges = {}, {}
    for name, (df_pert, df_base) in slices.items():
        D = compute_dynamic_pressure_D(df_pert) if color_mode != ABSOLUE and option == "B" else None
        errors[name] = pointwise_error(
            df_pert[f"{force_col}_pred"].values, df_base[f"{force_col}_pred"].values,
            color_mode, option, is_ft, D,
        )
        r_ranges[name] = (df_pert["r"].min(), df_pert["r"].max())
    vmax_of = _vmax_lookup(errors, shared_scale)

    fig = Figure(figsize=(7 * len(slices), 6))
    for i, (name, (df_pert, _)) in enumerate(slices.items()):
        vmax = vmax_of(name)
        norm = _norm(vmax, log_scale)
        ax = fig.add_subplot(1, len(slices), i + 1, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        theta_rad = np.deg2rad(df_pert["theta"].values)
        err_vals = _clip_for_log(errors[name], log_scale, vmax)
        sc = ax.scatter(theta_rad, df_pert["r"].values, c=err_vals, cmap="jet", s=40, norm=norm)
        _draw_neighborhood_outline(ax, axis, bounds, *r_ranges[name])
        _draw_center_line(ax, axis, center_value, *r_ranges[name])
        unit = "%" if color_mode != ABSOLUE else "N/m"
        ax.set_title(f"{name}", pad=20, fontweight="bold")
        fig.colorbar(sc, ax=ax, label=f"{color_mode} {force_col} ({unit})")
    fig.tight_layout()
    return fig


def envelope_scatter_figure(data: dict, coef: str, color_mode: str, log_scale: bool = False, shared_scale: bool = True):
    """data : {nom_modele: {'train_used': df, 'train_unused': df, 'test': df}} avec colonnes yaw,
    TSR, {coef}_abs_err, {coef}_rel_err. Renvoie une figure scatter :
    - carré = points d'entraînement effectivement utilisés par le modèle (sous-échantillonnage _P{pct})
    - triangle = points d'entraînement disponibles mais non utilisés par ce modèle
    - étoile = points de test
    `shared_scale` : True = échelle de couleur commune à tous les sous-graphiques, False = propre.
    """
    err_col = f"{coef}_abs_err" if color_mode == ABSOLUE else f"{coef}_rel_err"
    per_model_vals = {
        name: np.concatenate([d[key][err_col].values for key in ("train_used", "train_unused", "test")])
        for name, d in data.items()
    }
    vmax_of = _vmax_lookup(per_model_vals, shared_scale)

    fig = Figure(figsize=(7 * len(data), 6))
    for i, (name, d) in enumerate(data.items()):
        vmax = vmax_of(name)
        norm = _norm(vmax, log_scale)
        ax = fig.add_subplot(1, len(data), i + 1)
        used, unused, te = d["train_used"], d["train_unused"], d["test"]

        if len(unused):
            ax.scatter(unused["yaw"], unused["TSR"], c=_clip_for_log(unused[err_col].values, log_scale, vmax),
                       marker="^", s=70, cmap="coolwarm", norm=norm)
        if len(used):
            sc = ax.scatter(used["yaw"], used["TSR"], c=_clip_for_log(used[err_col].values, log_scale, vmax),
                            marker="s", s=70, cmap="coolwarm", norm=norm)
        else:
            sc = ax.scatter([], [], c=[], marker="s", s=70, cmap="coolwarm", norm=norm)
        ax.scatter(te["yaw"], te["TSR"], c=_clip_for_log(te[err_col].values, log_scale, vmax),
                   marker="*", s=140, cmap="coolwarm", norm=norm, edgecolor="black")

        unit = "" if color_mode == ABSOLUE else "%"
        train_mean = _train_mean(used, unused, err_col)
        train_used_mean = _train_used_mean(used, err_col)
        ax.set_title(
            f"{name}\n{coef} — Moyenne Train: {train_mean:.3g}{unit} | Train U: {train_used_mean:.3g}{unit} | "
            f"Test: {te[err_col].mean():.3g}{unit}",
            fontweight="bold",
        )
        ax.set_xlabel("Yaw (°)")
        ax.set_ylabel("TSR")
        handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Train (utilisé)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="Train (non utilisé)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=15, label="Test"),
        ]
        ax.legend(handles=handles)
        fig.colorbar(sc, ax=ax, label=f"{color_mode} {coef} ({unit})" if unit else f"{color_mode} {coef}")
    fig.tight_layout()
    return fig


def _train_mean(used, unused, err_col):
    """Moyenne de err_col sur l'union train_used + train_unused (l'un des deux peut être vide,
    ex. tous les points sont "used" pour un baseline sans sous-échantillonnage)."""
    parts = [d[err_col] for d in (used, unused) if len(d)]
    if not parts:
        return float("nan")
    return float(pd.concat(parts).mean())


def _train_used_mean(used, err_col):
    """Moyenne de err_col sur les seuls points de train effectivement utilisés à l'entraînement
    (sous-échantillonnage _P{pct}, ou totalité des points train pour un baseline BEM)."""
    return float(used[err_col].mean()) if len(used) else float("nan")


def score_scatter_figure(data: dict, option: str, log_scale: bool = False, shared_scale: bool = True):
    """data : {nom_modele: {'train_used': df, 'train_unused': df, 'test': df}} avec colonnes yaw,
    TSR, score (Score A ou B local (%), cf. metrics.local_relative_score, agrégé sur la grille
    (r, theta) de chaque couple yaw, TSR). Même forme que envelope_scatter_figure : carré = train
    utilisé, triangle = train non utilisé, étoile = test. `shared_scale` : True = échelle de
    couleur commune à tous les sous-graphiques, False = propre.
    """
    err_col = "score"
    per_model_vals = {
        name: np.concatenate([d[key][err_col].values for key in ("train_used", "train_unused", "test")])
        for name, d in data.items()
    }
    vmax_of = _vmax_lookup(per_model_vals, shared_scale)

    fig = Figure(figsize=(7 * len(data), 6))
    for i, (name, d) in enumerate(data.items()):
        vmax = vmax_of(name)
        norm = _norm(vmax, log_scale)
        ax = fig.add_subplot(1, len(data), i + 1)
        used, unused, te = d["train_used"], d["train_unused"], d["test"]

        if len(unused):
            ax.scatter(unused["yaw"], unused["TSR"], c=_clip_for_log(unused[err_col].values, log_scale, vmax),
                       marker="^", s=70, cmap="coolwarm", norm=norm)
        if len(used):
            sc = ax.scatter(used["yaw"], used["TSR"], c=_clip_for_log(used[err_col].values, log_scale, vmax),
                            marker="s", s=70, cmap="coolwarm", norm=norm)
        else:
            sc = ax.scatter([], [], c=[], marker="s", s=70, cmap="coolwarm", norm=norm)
        ax.scatter(te["yaw"], te["TSR"], c=_clip_for_log(te[err_col].values, log_scale, vmax),
                   marker="*", s=140, cmap="coolwarm", norm=norm, edgecolor="black")

        train_mean = _train_mean(used, unused, err_col)
        train_used_mean = _train_used_mean(used, err_col)
        ax.set_title(
            f"{name}\nScore {option} — Moyenne Train: {train_mean:.3g}% | Train U: {train_used_mean:.3g}% | "
            f"Test: {te[err_col].mean():.3g}%",
            fontweight="bold",
        )
        ax.set_xlabel("Yaw (°)")
        ax.set_ylabel("TSR")
        handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Train (utilisé)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="Train (non utilisé)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=15, label="Test"),
        ]
        ax.legend(handles=handles)
        fig.colorbar(sc, ax=ax, label=f"Score {option} (%)")
    fig.tight_layout()
    return fig


def uniform_curve_figure(x, series: dict, xlabel: str, ylabel: str, title: str):
    """series : {label: y_values (même longueur que x)}. Trace une courbe par série."""
    fig = Figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, marker=markers[i % len(markers)], lw=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    return fig
