import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st

from app.core_app import bem_provider, metrics, plotting
from app.core_app.data_access import load_train_test
from app.core_app.inference import predict_model
from app.core_app.models_registry import (
    OPTION_DESCRIPTIONS,
    describe_model,
    list_available_models,
    needs_bem,
)

st.set_page_config(page_title="Test LHS", layout="wide")
st.title("Test LHS")

NS = "lhs"  # namespace des clés de session_state pour cette page


def _key(name):
    return f"{NS}_{name}"


for k, default in [("model1", None), ("model2", None), ("option", "A"), ("preds", None), ("exec_time", None)]:
    st.session_state.setdefault(_key(k), default)

sous_mode = st.radio(
    "Sous-mode",
    ["Paramètres", "Carte F_n", "Carte F_t", "Enveloppe C_P", "Enveloppe C_T"],
    horizontal=True,
    key=_key("sous_mode"),
)

models = list_available_models()

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

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        df_train, df_test = load_train_test()

        selected = [model1, model2]
        if any(needs_bem(m) for m in selected):
            with st.spinner("Calcul de la BEM via bemol (train + test)…"):
                df_train = bem_provider.attach_bem_columns(df_train)
                df_test = bem_provider.attach_bem_columns(df_test)

        preds = {"train": {}, "test": {}}
        try:
            with st.spinner("Inférence des modèles…"):
                for m in selected:
                    preds["train"][m] = predict_model(df_train, m)
                    preds["test"][m] = predict_model(df_test, m)
        except Exception as exc:
            st.error(f"Échec de l'inférence : {exc}")
            st.stop()

        st.session_state[_key("preds")] = preds
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
# SOUS-MODES NÉCESSITANT DES PRÉDICTIONS DÉJÀ CALCULÉES
# =============================================================================
else:
    preds = st.session_state[_key("preds")]
    if preds is None:
        st.warning("Calculez d'abord les modèles dans le sous-mode Paramètres.")
        st.stop()

    model1, model2 = st.session_state[_key("model1")], st.session_state[_key("model2")]
    option = st.session_state[_key("option")]
    df_test_m1 = preds["test"][model1]

    if sous_mode in ("Carte F_n", "Carte F_t"):
        force_col = "Fn" if sous_mode == "Carte F_n" else "Ft"

        pairs = df_test_m1[["yaw", "TSR"]].drop_duplicates().reset_index(drop=True)
        labels = [f"yaw={row.yaw:.2f}° / TSR={row.TSR:.2f}" for row in pairs.itertuples()]

        def _pick_random_pair():
            st.session_state[_key("pair_idx")] = int(np.random.randint(len(labels)))

        c1, c2 = st.columns([3, 1])
        with c1:
            idx = st.selectbox("Couple (yaw, TSR) du test", range(len(labels)), format_func=lambda i: labels[i], key=_key("pair_idx"))
        with c2:
            st.button("Point aléatoire", key=_key("pair_random"), on_click=_pick_random_pair)

        color_mode = st.radio("Coloration", [metrics.ABSOLUE, metrics.RELATIVE], horizontal=True, key=_key("color_mode"))

        yaw_sel, tsr_sel = pairs.loc[idx, "yaw"], pairs.loc[idx, "TSR"]
        slices = {}
        for m in (model1, model2):
            df_m = preds["test"][m]
            slices[m] = df_m[(df_m["yaw"] == yaw_sel) & (df_m["TSR"] == tsr_sel)]

        fig = plotting.polar_error_figure(slices, force_col, color_mode, option)
        st.pyplot(fig)

    elif sous_mode in ("Enveloppe C_P", "Enveloppe C_T"):
        coef = "Cp" if sous_mode == "Enveloppe C_P" else "Ct"
        color_mode = st.radio("Coloration", [metrics.ABSOLUE, metrics.RELATIVE], horizontal=True, key=_key("color_mode_env"))

        data = {}
        for m in (model1, model2):
            data[m] = {
                "train": metrics.cp_ct_table(preds["train"][m], "pred"),
                "test": metrics.cp_ct_table(preds["test"][m], "pred"),
            }
        fig = plotting.envelope_scatter_figure(data, coef, color_mode)
        st.pyplot(fig)
