"""Fonctions de tracé (matplotlib) réutilisées par les pages Test LHS / Test Uniforme."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from app.core_app.metrics import ABSOLUE, local_relative_score, pointwise_error
from core.physics import compute_dynamic_pressure_D


def polar_error_figure(slices: dict, force_col: str, color_mode: str, option: str):
    """slices : {nom_modele: df_filtré_sur_(yaw,TSR)} avec colonnes r, theta,
    {force_col}_pred, {force_col}_SVEN. Renvoie une figure avec une polaire par modèle,
    échelle de couleur partagée."""
    is_ft = force_col == "Ft"
    errors = {}
    for name, df in slices.items():
        D = compute_dynamic_pressure_D(df) if color_mode != ABSOLUE and option == "B" else None
        errors[name] = pointwise_error(
            df[f"{force_col}_pred"].values, df[f"{force_col}_SVEN"].values, color_mode, option, is_ft, D,
        )
    vmax = max(err.max() for err in errors.values()) if errors else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig = Figure(figsize=(7 * len(slices), 6))
    for i, (name, df) in enumerate(slices.items()):
        ax = fig.add_subplot(1, len(slices), i + 1, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        theta_rad = np.deg2rad(df["theta"].values)
        sc = ax.scatter(theta_rad, df["r"].values, c=errors[name], cmap="jet", s=40, vmin=0, vmax=vmax)
        score = local_relative_score(df, option)
        unit = "%" if color_mode != ABSOLUE else "N/m"
        cbar_label = f"{color_mode} {force_col} ({unit})"
        ax.set_title(f"{name}\nScore {option} local ({force_col.lower()}, r-theta) : {score:.2f}%", pad=20, fontweight="bold")
        fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    return fig


def envelope_scatter_figure(data: dict, coef: str, color_mode: str):
    """data : {nom_modele: {'train': df_scores, 'test': df_scores}} avec colonnes yaw, TSR,
    {coef}_abs_err, {coef}_rel_err. Renvoie une figure scatter (rond=train, étoile=test)."""
    err_col = f"{coef}_abs_err" if color_mode == ABSOLUE else f"{coef}_rel_err"
    all_vals = []
    for d in data.values():
        all_vals.append(d["train"][err_col].values)
        all_vals.append(d["test"][err_col].values)
    vmax = max(v.max() for v in all_vals if len(v)) if all_vals else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig = Figure(figsize=(7 * len(data), 6))
    for i, (name, d) in enumerate(data.items()):
        ax = fig.add_subplot(1, len(data), i + 1)
        tr, te = d["train"], d["test"]
        sc = ax.scatter(tr["yaw"], tr["TSR"], c=tr[err_col], marker="o", s=70, cmap="coolwarm", vmin=0, vmax=vmax)
        ax.scatter(te["yaw"], te["TSR"], c=te[err_col], marker="*", s=140, cmap="coolwarm", vmin=0, vmax=vmax, edgecolor="black")
        unit = "" if color_mode == ABSOLUE else "%"
        ax.set_title(f"{name}\n{coef} — Moyenne Train: {tr[err_col].mean():.3g}{unit} | Test: {te[err_col].mean():.3g}{unit}", fontweight="bold")
        ax.set_xlabel("Yaw (°)")
        ax.set_ylabel("TSR")
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Train"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=15, label="Test"),
        ]
        ax.legend(handles=handles)
        fig.colorbar(sc, ax=ax, label=f"{color_mode} {coef} ({unit})" if unit else f"{color_mode} {coef}")
    fig.tight_layout()
    return fig


def uniform_curve_figure(x, series: dict, xlabel: str, ylabel: str, title: str):
    """series : {label: y_values (même longueur que x)}. Trace une courbe par série."""
    fig = Figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, marker=markers[i % len(markers)], lw=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    return fig
