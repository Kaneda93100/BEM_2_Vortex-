import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.core_app import metrics, perturbation, plotting
from app.core_app.data_access import load_train_test
from app.core_app.inference import predict_model
from app.core_app.models_registry import (
    OPTION_DESCRIPTIONS,
    describe_model,
    model_bem_suffix,
    models_using_bem,
    parse_model_name,
)
from core.config import Dist_R

st.set_page_config(page_title="Corrélation", layout="wide")
st.title("Corrélation")
st.caption(
    "Étude de sensibilité : on perturbe localement (gaussienne centrée sur un r ou theta choisi) "
    "l'entrée BEM d'un modèle utilisant la résiduelle '1', '2' ou '2+', et on observe l'écart de "
    "prédiction par rapport à la prédiction non perturbée, sur un couple (yaw, TSR) du dataset LHS "
)

NS = "corr"


def _key(name):
    return f"{NS}_{name}"


R_GRID = sorted(Dist_R)
THETA_GRID = [float(t) for t in np.arange(0, 360, 5.0)]
NEIGHBORHOOD_CHOICES = {"r": list(range(0, 13, 2)), "theta": list(range(0, 25, 4))}

for k, default in [
    ("model1", None), ("model2", None), ("option", "A"), ("axis", "r"), ("bounds", None),
    ("center_value", None), ("results", None), ("exec_time", None),
]:
    st.session_state.setdefault(_key(k), default)

sous_mode = st.radio(
    "Sous-mode",
    ["Paramètres", "F_n sur F_n", "F_n sur F_t", "F_t sur F_n", "F_t sur F_t"],
    horizontal=True,
    key=_key("sous_mode"),
)

df_train, df_test = load_train_test()
df_lhs = pd.concat([df_train, df_test], ignore_index=True)
lhs_pairs = df_lhs[["yaw", "TSR"]].drop_duplicates().reset_index(drop=True)
lhs_labels = [f"yaw={row.yaw:.2f}° / TSR={row.TSR:.2f}" for row in lhs_pairs.itertuples()]

# =============================================================================
# PARAMÈTRES
# =============================================================================
if sous_mode == "Paramètres":
    option = st.radio("Métrique (Score)", ["A", "B"], horizontal=True, key=_key("option_select"))
    with st.expander("Différence entre Score A et Score B"):
        st.markdown(f"**Score A** : {OPTION_DESCRIPTIONS['A']}")
        st.markdown(f"**Score B** : {OPTION_DESCRIPTIONS['B']}")

    models = models_using_bem(option)
    if len(models) < 2:
        st.error("Moins de 2 modèles résiduelle '1'/'2'/'2+' disponibles pour ce score.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Modèle 1", models, key=_key("model1_select"))
        st.caption(describe_model(model1))
    with col2:
        model2 = st.selectbox("Modèle 2", models, index=min(1, len(models) - 1), key=_key("model2_select"))
        st.caption(describe_model(model2))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        axis = st.radio("Axe de la perturbation", ["r", "theta"], horizontal=True, key=_key("axis_select"))
    with c2:
        st.caption("Perturbation gaussienne centrée sur la valeur choisie, appliquée à tous les "
                    "points de l'autre coordonnée (tout theta si axe=r, tout r si axe=theta).")

    def _pick_random_r():
        st.session_state[_key("value_r")] = float(np.random.choice(R_GRID))

    def _pick_random_theta():
        st.session_state[_key("value_theta")] = float(np.random.choice(THETA_GRID))

    cv1, cv2 = st.columns([3, 1])
    if axis == "r":
        with cv1:
            value = st.selectbox("Valeur de r (m)", R_GRID, format_func=lambda v: f"{v:.4f}", key=_key("value_r"))
        with cv2:
            st.write("")
            st.button("Aléatoire", key=_key("value_r_random"), on_click=_pick_random_r)
    else:
        with cv1:
            value = st.selectbox("Valeur de theta (°)", THETA_GRID, format_func=lambda v: f"{v:.0f}", key=_key("value_theta"))
        with cv2:
            st.write("")
            st.button("Aléatoire", key=_key("value_theta_random"), on_click=_pick_random_theta)

    neighborhood = st.select_slider(
        "Voisinage (nombre pair de positions de grille)",
        options=NEIGHBORHOOD_CHOICES[axis], value=NEIGHBORHOOD_CHOICES[axis][len(NEIGHBORHOOD_CHOICES[axis]) // 2],
        key=_key(f"neighborhood_{axis}"),
    )
    intensity = st.slider("Intensité relative de la perturbation (%)", 50, 500, 100, step=10, key=_key("intensity"))
    st.caption(
        f"Au centre choisi, la valeur BEM perturbée vaudra {intensity}% de sa valeur originale "
        f"(50% = divisée par 2, 500% = multipliée par 5) ; loin du centre en pointillés, elle reste inchangée."
    )

    st.divider()
    c3, c4 = st.columns([3, 1])
    with c3:
        pair_idx = st.selectbox("Couple (yaw, TSR) du dataset LHS", range(len(lhs_labels)), format_func=lambda i: lhs_labels[i], key=_key("pair_idx"))
    with c4:
        st.write("")

        def _pick_random_pair():
            st.session_state[_key("pair_idx")] = int(np.random.randint(len(lhs_labels)))

        st.button("Point aléatoire", key=_key("pair_random"), on_click=_pick_random_pair)

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        yaw_sel, tsr_sel = lhs_pairs.loc[pair_idx, "yaw"], lhs_pairs.loc[pair_idx, "TSR"]
        base_df = df_lhs[(df_lhs["yaw"] == yaw_sel) & (df_lhs["TSR"] == tsr_sel)].reset_index(drop=True)

        results = {}
        try:
            for m in (model1, model2):
                info = parse_model_name(m)
                inter, bem_suffix = info["inter"], model_bem_suffix(m)

                baseline = predict_model(base_df, m)
                pert_fn_df = perturbation.perturb_bem(base_df, bem_suffix, inter, "Fn", axis, value, neighborhood, intensity)
                pert_ft_df = perturbation.perturb_bem(base_df, bem_suffix, inter, "Ft", axis, value, neighborhood, intensity)
                results[m] = {
                    "baseline": baseline,
                    "pert_Fn": predict_model(pert_fn_df, m),
                    "pert_Ft": predict_model(pert_ft_df, m),
                }
        except Exception as exc:
            st.error(f"Échec du calcul : {exc}")
            st.stop()

        bounds = perturbation.neighborhood_bounds(base_df, axis, value, neighborhood)

        st.session_state[_key("results")] = results
        st.session_state[_key("model1")] = model1
        st.session_state[_key("model2")] = model2
        st.session_state[_key("option")] = option
        st.session_state[_key("axis")] = axis
        st.session_state[_key("bounds")] = bounds
        st.session_state[_key("center_value")] = value
        st.session_state[_key("exec_time")] = time.perf_counter() - t0

    if st.session_state[_key("exec_time")] is not None:
        st.success(f"Calcul terminé en {st.session_state[_key('exec_time')]:.2f} s.")
    else:
        st.info("Choisissez les paramètres puis cliquez sur Calculer.")

# =============================================================================
# SOUS-MODES DE RÉSULTATS (cartes polaires)
# =============================================================================
else:
    results = st.session_state[_key("results")]
    if results is None:
        st.warning("Calculez d'abord la perturbation dans le sous-mode Paramètres.")
        st.stop()

    model1, model2 = st.session_state[_key("model1")], st.session_state[_key("model2")]
    option, axis, bounds = st.session_state[_key("option")], st.session_state[_key("axis")], st.session_state[_key("bounds")]
    center_value = st.session_state[_key("center_value")]

    pert_component, force_col = {
        "F_n sur F_n": ("pert_Fn", "Fn"),
        "F_n sur F_t": ("pert_Fn", "Ft"),
        "F_t sur F_n": ("pert_Ft", "Fn"),
        "F_t sur F_t": ("pert_Ft", "Ft"),
    }[sous_mode]

    st.caption(
        f"Perturbation sur l'entrée {'F_n (ou a_n·V_app)' if pert_component == 'pert_Fn' else 'F_t (ou a_t·V_app)'} "
        f"BEM, effet observé sur la prédiction {force_col} — écart par rapport à la prédiction non perturbée."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        color_mode = st.radio("Coloration", [metrics.ABSOLUE, metrics.RELATIVE], horizontal=True, key=_key("color_mode"))
    with c2:
        log_scale = st.radio("Échelle des couleurs", ["Linéaire", "Logarithmique"], horizontal=True, key=_key("log_scale")) == "Logarithmique"
    with c3:
        shared_scale = st.radio("Échelle", [plotting.SCALE_COMMUNE, plotting.SCALE_PROPRE], horizontal=True, key=_key("scale_mode")) == plotting.SCALE_COMMUNE

    slices = {m: (results[m][pert_component], results[m]["baseline"]) for m in (model1, model2)}
    fig = plotting.polar_perturbation_figure(
        slices, force_col, color_mode, option, axis, bounds, center_value,
        log_scale=log_scale, shared_scale=shared_scale,
    )
    st.pyplot(fig)
