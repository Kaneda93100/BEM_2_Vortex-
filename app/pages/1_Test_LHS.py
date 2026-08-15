import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.core_app import metrics, plotting
from app.core_app.data_access import load_train_test
from app.core_app.inference import predict_model_chunked
from app.core_app.models_registry import (
    OPTION_DESCRIPTIONS,
    describe_model,
    is_baseline,
    models_for_option,
    parse_model_name,
)
from core.config import RANDOM_SEED
from training.src.data_loader import subsample_train

st.set_page_config(page_title="Test LHS", layout="wide")
st.title("Test LHS")

NS = "lhs"  # namespace des clés de session_state pour cette page


def _key(name):
    return f"{NS}_{name}"


for k, default in [("model1", None), ("model2", None), ("option", "A"), ("preds", None), ("exec_time", None)]:
    st.session_state.setdefault(_key(k), default)


def _used_unused_train(df_train_pred, model_name):
    """Scinde les lignes train (déjà prédites) d'un modèle entre points effectivement utilisés à
    l'entraînement (sous-échantillonnage _P{pct}) et points disponibles mais non utilisés. Un
    baseline BEM (pas de notion d'entraînement) est considéré comme "utilisant" tous les points."""
    if is_baseline(model_name):
        used_yaws = set(df_train_pred["yaw"].unique())
    else:
        pct = parse_model_name(model_name)["pct"]
        used_yaws = set(subsample_train(df_train_pred, pct, seed=RANDOM_SEED)["yaw"].unique())
    mask = df_train_pred["yaw"].isin(used_yaws)
    return df_train_pred[mask], df_train_pred[~mask]


_CP_CT_COLUMNS = ["yaw", "TSR", "Cp_pred", "Ct_pred", "Cp_true", "Ct_true", "Cp_abs_err", "Ct_abs_err", "Cp_rel_err", "Ct_rel_err"]


def _cp_ct_or_empty(df, pred_prefix):
    if len(df) == 0:
        return pd.DataFrame(columns=_CP_CT_COLUMNS)
    return metrics.cp_ct_table(df, pred_prefix)


sous_mode = st.radio(
    "Sous-mode",
    ["Paramètres", "Carte F_n", "Carte F_t", "Enveloppe C_P", "Enveloppe C_T"],
    horizontal=True,
    key=_key("sous_mode"),
)

# =============================================================================
# PARAMÈTRES
# =============================================================================
if sous_mode == "Paramètres":
    option = st.radio("Métrique (Score)", ["A", "B"], horizontal=True, key=_key("option_select"))
    with st.expander("Différence entre Score A et Score B"):
        st.markdown(f"**Score A** : {OPTION_DESCRIPTIONS['A']}")
        st.markdown(f"**Score B** : {OPTION_DESCRIPTIONS['B']}")
    st.caption(
        "Seuls les modèles entraînés avec cette option (+ les baselines BEM, compatibles avec les "
        "deux scores) sont proposés ci-dessous."
    )

    models = models_for_option(option)
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Modèle 1", models, key=_key("model1_select"))
        st.caption(describe_model(model1))
    with col2:
        model2 = st.selectbox("Modèle 2", models, index=min(1, len(models) - 1), key=_key("model2_select"))
        st.caption(describe_model(model2))

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        df_train, df_test = load_train_test()

        selected = [model1, model2]
        steps = [(split, m) for split in ("train", "test") for m in selected]
        n_pairs = {
            "train": df_train[["yaw", "TSR"]].drop_duplicates().shape[0],
            "test": df_test[["yaw", "TSR"]].drop_duplicates().shape[0],
        }
        grand_total = sum(n_pairs[split] for split, _ in steps)

        progress = st.progress(0.0, text="Inférence des modèles…")
        preds = {"train": {}, "test": {}}
        done_before = 0
        try:
            for split, m in steps:
                df_in = df_train if split == "train" else df_test

                def _cb(done, total, _base=done_before):
                    progress.progress((_base + done) / grand_total, text=f"Inférence des modèles… ({_base + done}/{grand_total} couples yaw,TSR)")

                preds[split][m] = predict_model_chunked(df_in, m, progress_cb=_cb)
                done_before += n_pairs[split]
        except Exception as exc:
            progress.empty()
            st.error(f"Échec de l'inférence : {exc}")
            st.stop()
        progress.empty()

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

        c3, c4, c5 = st.columns(3)
        with c3:
            color_mode = st.radio("Coloration", [metrics.ABSOLUE, metrics.RELATIVE], horizontal=True, key=_key("color_mode"))
        with c4:
            log_scale = st.radio("Échelle des couleurs", ["Linéaire", "Logarithmique"], horizontal=True, key=_key("log_scale")) == "Logarithmique"
        with c5:
            shared_scale = st.radio("Échelle", [plotting.SCALE_COMMUNE, plotting.SCALE_PROPRE], horizontal=True, key=_key("scale_mode")) == plotting.SCALE_COMMUNE

        yaw_sel, tsr_sel = pairs.loc[idx, "yaw"], pairs.loc[idx, "TSR"]
        slices = {}
        for m in (model1, model2):
            df_m = preds["test"][m]
            slices[m] = df_m[(df_m["yaw"] == yaw_sel) & (df_m["TSR"] == tsr_sel)]

        fig = plotting.polar_error_figure(slices, force_col, color_mode, option, log_scale=log_scale, shared_scale=shared_scale)
        st.pyplot(fig)

    elif sous_mode in ("Enveloppe C_P", "Enveloppe C_T"):
        coef = "Cp" if sous_mode == "Enveloppe C_P" else "Ct"
        c1, c2, c3 = st.columns(3)
        with c1:
            color_mode = st.radio("Coloration", [metrics.ABSOLUE, metrics.RELATIVE], horizontal=True, key=_key("color_mode_env"))
        with c2:
            log_scale = st.radio("Échelle des couleurs", ["Linéaire", "Logarithmique"], horizontal=True, key=_key("log_scale_env")) == "Logarithmique"
        with c3:
            shared_scale = st.radio("Échelle", [plotting.SCALE_COMMUNE, plotting.SCALE_PROPRE], horizontal=True, key=_key("scale_mode_env")) == plotting.SCALE_COMMUNE

        data = {}
        for m in (model1, model2):
            train_used, train_unused = _used_unused_train(preds["train"][m], m)
            data[m] = {
                "train_used": _cp_ct_or_empty(train_used, "pred"),
                "train_unused": _cp_ct_or_empty(train_unused, "pred"),
                "test": _cp_ct_or_empty(preds["test"][m], "pred"),
            }
        fig = plotting.envelope_scatter_figure(data, coef, color_mode, log_scale=log_scale, shared_scale=shared_scale)
        st.pyplot(fig)
