import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st

from app.core_app import bem_provider, plotting
from app.core_app.data_access import load_castor_sven
from app.core_app.inference import predict_model
from app.core_app.models_registry import (
    OPTION_DESCRIPTIONS,
    describe_model,
    list_available_models,
)
from core.physics import compute_cp

st.set_page_config(page_title="Test Uniforme", layout="wide")
st.title("Test Uniforme")
st.caption(
    "Basé sur data/castor_sven/forces_BEM_SVEN_CASTOR.csv — actuellement TSR fixe, yaw balayé "
    "sur les valeurs disponibles dans ce fichier (plus de TSR seront ajoutés ultérieurement)."
)

NS = "uni"


def _key(name):
    return f"{NS}_{name}"


for k, default in [("model1", None), ("model2", None), ("option", "A"), ("grid_preds", None), ("exec_time", None)]:
    st.session_state.setdefault(_key(k), default)

sous_mode = st.radio(
    "Sous-mode",
    ["Paramètres", "Carte F_n", "Carte F_t", "Enveloppe C_P", "Enveloppe C_T"],
    horizontal=True,
    key=_key("sous_mode"),
)

models = list_available_models()
castor_df = load_castor_sven()
yaws_dispo = sorted(castor_df["yaw"].unique())
tsr_fixe = float(castor_df["TSR"].iloc[0])

# =============================================================================
# PARAMÈTRES
# =============================================================================
if sous_mode == "Paramètres":
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Modèle 1", models, key=_key("model1_select"))
        st.caption(describe_model(model1))
    with col2:
        model2 = st.selectbox("Modèle 2", models, index=min(1, len(models) - 1), key=_key("model2_select"))
        st.caption(describe_model(model2))

    option = st.radio("Métrique (Score)", ["A", "B"], horizontal=True, key=_key("option_select"))
    with st.expander("Différence entre Score A et Score B"):
        st.markdown(f"**Score A** : {OPTION_DESCRIPTIONS['A']}")
        st.markdown(f"**Score B** : {OPTION_DESCRIPTIONS['B']}")

    st.caption(f"Yaw balayés : {yaws_dispo}° — TSR fixe : {tsr_fixe}")

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        pairs = [(y, tsr_fixe) for y in yaws_dispo]
        with st.spinner("Calcul de la BEM via bemol (grille native 36x72)…"):
            grid_df = bem_provider.compute_bem(pairs, nbr_az=72)

        preds = {}
        try:
            with st.spinner("Inférence des modèles…"):
                for m in (model1, model2):
                    preds[m] = predict_model(grid_df, m)
        except Exception as exc:
            st.error(f"Échec de l'inférence : {exc}")
            st.stop()

        st.session_state[_key("grid_preds")] = preds
        st.session_state[_key("model1")] = model1
        st.session_state[_key("model2")] = model2
        st.session_state[_key("option")] = option
        st.session_state[_key("exec_time")] = time.perf_counter() - t0

    if st.session_state[_key("exec_time")] is not None:
        st.success(f"Calcul terminé en {st.session_state[_key('exec_time')]:.2f} s.")
        st.caption(
            f"Modèle 1 : {st.session_state[_key('model1')]} — "
            f"Modèle 2 : {st.session_state[_key('model2')]} — "
            f"Score {st.session_state[_key('option')]}"
        )
    else:
        st.info("Choisissez deux modèles puis cliquez sur Calculer.")

# =============================================================================
# SOUS-MODES NÉCESSITANT LA GRILLE DÉJÀ CALCULÉE
# =============================================================================
else:
    grid_preds = st.session_state[_key("grid_preds")]
    if grid_preds is None:
        st.warning("Calculez d'abord les modèles dans le sous-mode Paramètres.")
        st.stop()

    model1, model2 = st.session_state[_key("model1")], st.session_state[_key("model2")]

    if sous_mode in ("Carte F_n", "Carte F_t"):
        force_col = "Fn" if sous_mode == "Carte F_n" else "Ft"

        r_options = sorted(castor_df["r"].unique())
        theta_options = sorted(castor_df["theta"].unique())

        def _pick_random_point():
            st.session_state[_key("r_sel")] = float(np.random.choice(r_options))
            st.session_state[_key("theta_sel")] = float(np.random.choice(theta_options))

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            r_sel = st.selectbox("Rayon r", r_options, format_func=lambda v: f"{v:.4f}", key=_key("r_sel"))
        with c2:
            theta_sel = st.selectbox("Azimut theta (°)", theta_options, format_func=lambda v: f"{v:.0f}", key=_key("theta_sel"))
        with c3:
            st.write("")
            st.write("")
            st.button("Point aléatoire", key=_key("point_random"), on_click=_pick_random_point)

        castor_pt = castor_df[np.isclose(castor_df["r"], r_sel) & np.isclose(castor_df["theta"], theta_sel)]
        castor_pt = castor_pt.set_index("yaw").reindex(yaws_dispo)

        cols = st.columns(2)
        for col, model_name in zip(cols, (model1, model2)):
            grid_df = grid_preds[model_name]
            grid_pt = grid_df[np.isclose(grid_df["r"], r_sel) & np.isclose(grid_df["theta"], theta_sel)]
            grid_pt = grid_pt.set_index("yaw").reindex(yaws_dispo)

            series = {
                "BEM baseline (CSV)": castor_pt[f"{force_col}_BEM"].values,
                model_name: grid_pt[f"{force_col}_pred"].values,
                "SVEN": castor_pt[f"{force_col}_SVEN"].values,
                "Castor": castor_pt[f"{force_col}_Castor"].values,
            }
            fig = plotting.uniform_curve_figure(
                yaws_dispo, series, "Yaw (°)", f"{force_col} (N/m)",
                f"{model_name}\nr={r_sel:.3f}, theta={theta_sel:.0f}°",
            )
            with col:
                st.pyplot(fig)

    elif sous_mode in ("Enveloppe C_P", "Enveloppe C_T"):
        coef = "Cp" if sous_mode == "Enveloppe C_P" else "Ct"

        bem_cp = compute_cp(castor_df, "Fn_BEM", "Ft_BEM").set_index("yaw").reindex(yaws_dispo)
        sven_cp = compute_cp(castor_df, "Fn_SVEN", "Ft_SVEN").set_index("yaw").reindex(yaws_dispo)
        castor_cp = compute_cp(castor_df, "Fn_Castor", "Ft_Castor").set_index("yaw").reindex(yaws_dispo)

        cols = st.columns(2)
        for col, model_name in zip(cols, (model1, model2)):
            model_cp = compute_cp(grid_preds[model_name], "Fn_pred", "Ft_pred").set_index("yaw").reindex(yaws_dispo)
            series = {
                "BEM baseline (CSV)": bem_cp[f"{coef}_BEM"].values,
                model_name: model_cp[f"{coef}_pred"].values,
                "SVEN": sven_cp[f"{coef}_SVEN"].values,
                "Castor": castor_cp[f"{coef}_Castor"].values,
            }
            fig = plotting.uniform_curve_figure(
                yaws_dispo, series, "Yaw (°)", coef, f"{model_name} — Enveloppe {coef}",
            )
            with col:
                st.pyplot(fig)
