import itertools
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st

from app.core_app import bem_provider
from app.core_app.inference import predict_model
from app.core_app.models_registry import (
    BASELINE_BEM,
    describe_model,
    list_available_models,
    parse_model_name,
)

st.set_page_config(page_title="Générateur", layout="wide")
st.title("Générateur de prédictions")

models = list_available_models()
model_name = st.selectbox("Modèle", models)
st.caption(describe_model(model_name))

has_v_fields = model_name == BASELINE_BEM or parse_model_name(model_name)["inter"] == "v"

EXPORT_COLS_BASE = ["yaw", "TSR", "r", "theta", "Fn_pred", "Ft_pred"]
EXPORT_COLS_V = EXPORT_COLS_BASE + ["V_eff_pred", "alpha_pred"]


def _export_columns():
    return EXPORT_COLS_V if has_v_fields else EXPORT_COLS_BASE


mode = st.radio("Mode", ["Ponctuel", "Uniforme"], horizontal=True)

if mode == "Ponctuel":
    c1, c2 = st.columns(2)
    with c1:
        yaw = st.number_input("Yaw (°)", value=10.0, step=1.0)
    with c2:
        tsr = st.number_input("TSR", value=8.0, step=0.5)

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        try:
            with st.spinner("Calcul BEM (bemol) + inférence…"):
                grid_df = bem_provider.compute_bem([(yaw, tsr)], nbr_az=72)
                df_pred = predict_model(grid_df, model_name)
        except Exception as exc:
            st.error(f"Échec du calcul : {exc}")
            st.stop()
        exec_time = time.perf_counter() - t0

        df_export = df_pred[_export_columns()].sort_values(["r", "theta"]).reset_index(drop=True)
        st.session_state["gen_point_export"] = df_export
        st.session_state["gen_point_time"] = exec_time

    if st.session_state.get("gen_point_export") is not None:
        st.success(f"Calcul terminé en {st.session_state['gen_point_time']:.2f} s. ({len(st.session_state['gen_point_export'])} lignes, grille 36×72)")
        st.dataframe(st.session_state["gen_point_export"].head(20))
        st.download_button(
            "Télécharger le CSV",
            st.session_state["gen_point_export"].to_csv(index=False).encode("utf-8"),
            file_name=f"generation_{model_name}_yaw{yaw:g}_tsr{tsr:g}.csv",
            mime="text/csv",
        )

else:
    c1, c2, c3 = st.columns(3)
    with c1:
        yaw_min = st.number_input("Yaw min (°)", value=5.0, step=1.0, key="gen_yaw_min")
        yaw_max = st.number_input("Yaw max (°)", value=30.0, step=1.0, key="gen_yaw_max")
        yaw_step = st.number_input("Pas yaw (°)", value=5.0, min_value=0.01, step=1.0, key="gen_yaw_step")
    with c2:
        tsr_min = st.number_input("TSR min", value=6.0, step=0.5, key="gen_tsr_min")
        tsr_max = st.number_input("TSR max", value=10.0, step=0.5, key="gen_tsr_max")
        tsr_step = st.number_input("Pas TSR", value=1.0, min_value=0.01, step=0.5, key="gen_tsr_step")
    with c3:
        yaws = np.arange(yaw_min, yaw_max + yaw_step / 2, yaw_step)
        tsrs = np.arange(tsr_min, tsr_max + tsr_step / 2, tsr_step)
        pairs = list(itertools.product(yaws, tsrs))
        st.metric("Couples (yaw, TSR)", len(pairs))
        st.metric("Lignes attendues (r, theta = 36×72)", len(pairs) * 36 * 72)

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        try:
            with st.spinner(f"Calcul BEM (bemol) + inférence sur {len(pairs)} couples (yaw, TSR)…"):
                grid_df = bem_provider.compute_bem(pairs, nbr_az=72)
                df_pred = predict_model(grid_df, model_name)
        except Exception as exc:
            st.error(f"Échec du calcul : {exc}")
            st.stop()
        exec_time = time.perf_counter() - t0

        df_export = df_pred[_export_columns()].sort_values(["yaw", "TSR", "r", "theta"]).reset_index(drop=True)
        st.session_state["gen_uniform_export"] = df_export
        st.session_state["gen_uniform_time"] = exec_time

    if st.session_state.get("gen_uniform_export") is not None:
        st.success(f"Calcul terminé en {st.session_state['gen_uniform_time']:.2f} s. ({len(st.session_state['gen_uniform_export'])} lignes)")
        st.dataframe(st.session_state["gen_uniform_export"].head(20))
        st.download_button(
            "Télécharger le CSV",
            st.session_state["gen_uniform_export"].to_csv(index=False).encode("utf-8"),
            file_name=f"generation_{model_name}_grille_uniforme.csv",
            mime="text/csv",
        )
