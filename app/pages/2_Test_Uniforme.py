import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st

from app.core_app import metrics, plotting
from app.core_app.data_access import load_bem_grid_cache, load_castor_sven
from app.core_app.inference import predict_model_chunked
from app.core_app.models_registry import (
    OPTION_DESCRIPTIONS,
    describe_model,
    models_for_option,
)
from core.config import BEM_SUFFIXES
from core.physics import compute_cp

# Sources de référence tracées en plus des 2 modèles comparés : les 3 variantes BEM (bemol), SVEN et Castor
REF_SOURCES = [(f"BEM ({s})", f"BEM_{s}") for s in BEM_SUFFIXES] + [("SVEN", "SVEN"), ("Castor", "Castor")]

st.set_page_config(page_title="Test Uniforme", layout="wide")
st.title("Test Uniforme")
st.caption(
    "Grille (yaw x TSR) de data/castor_sven/forces_BEM_SVEN_CASTOR.csv (4 TSR x 6 yaw). La BEM "
    "(3 variantes DUM/IFP/P&P) est lue depuis le cache précalculé "
)

NS = "uni"


def _key(name):
    return f"{NS}_{name}"


for k, default in [
    ("model1", None), ("model2", None), ("option", "A"),
    ("grid_df", None), ("grid_preds", None), ("exec_time", None), ("bem_meta", None),
]:
    st.session_state.setdefault(_key(k), default)

sous_mode = st.radio(
    "Sous-mode",
    ["Paramètres", "Carte F_n", "Carte F_t", "Enveloppe C_P", "Enveloppe C_T"],
    horizontal=True,
    key=_key("sous_mode"),
)

try:
    bem_df, bem_meta = load_bem_grid_cache()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
castor_df = load_castor_sven()

yaws_dispo = sorted(float(y) for y in bem_df["yaw"].unique())
tsrs_dispo = sorted(float(t) for t in bem_df["TSR"].unique())
pairs = list(bem_df[["yaw", "TSR"]].drop_duplicates().itertuples(index=False, name=None))

# =============================================================================
# PARAMÈTRES
# =============================================================================
if sous_mode == "Paramètres":
    option = st.radio("Métrique (Score)", ["A", "B"], horizontal=True, key=_key("option_select"))
    with st.expander("Différence entre Score A et Score B"):
        st.markdown(f"**Score A** : {OPTION_DESCRIPTIONS['A']}")
        st.markdown(f"**Score B** : {OPTION_DESCRIPTIONS['B']}")

    models = models_for_option(option)
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Modèle 1", models, key=_key("model1_select"))
        st.caption(describe_model(model1))
    with col2:
        model2 = st.selectbox("Modèle 2", models, index=min(1, len(models) - 1), key=_key("model2_select"))
        st.caption(describe_model(model2))

    st.caption(
        f"Grille : {len(pairs)} couples (yaw, TSR) — yaw {yaws_dispo}°, TSR {tsrs_dispo}. "
        f"BEM précalculée en {bem_meta['seconds']:.2f} s."
    )

    if st.button("Calculer", type="primary"):
        t0 = time.perf_counter()
        selected = [model1, model2]
        grand_total = len(pairs) * len(selected)

        infer_progress = st.progress(0.0, text="Inférence des modèles…")
        preds = {}
        done_before = 0
        try:
            for m in selected:
                def _cb(done, total, _base=done_before):
                    infer_progress.progress((_base + done) / grand_total, text=f"Inférence des modèles… ({_base + done}/{grand_total} couples yaw,TSR)")

                preds[m] = predict_model_chunked(bem_df, m, progress_cb=_cb)
                done_before += len(pairs)
        except Exception as exc:
            infer_progress.empty()
            st.error(f"Échec de l'inférence : {exc}")
            st.stop()
        infer_progress.empty()

        st.session_state[_key("grid_df")] = bem_df
        st.session_state[_key("grid_preds")] = preds
        st.session_state[_key("model1")] = model1
        st.session_state[_key("model2")] = model2
        st.session_state[_key("option")] = option
        st.session_state[_key("bem_meta")] = bem_meta
        st.session_state[_key("exec_time")] = time.perf_counter() - t0

    if st.session_state[_key("exec_time")] is not None:
        st.success(f"Inférence terminée en {st.session_state[_key('exec_time')]:.2f} s (BEM déjà en cache).")
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

    def _pick_random_tsr(state_key):
        st.session_state[state_key] = float(np.random.choice(tsrs_dispo))

    def _pick_random_yaw(state_key):
        st.session_state[state_key] = float(np.random.choice(yaws_dispo))

    if sous_mode in ("Carte F_n", "Carte F_t"):
        force_col = "Fn" if sous_mode == "Carte F_n" else "Ft"

        grandeur = st.radio(
            "Grandeur", ["Effort max |.| (N/m)", "Rayon du max (m)", "Azimut du max (°)"],
            horizontal=True, key=_key(f"grandeur_{force_col}"),
        )
        value_col = {
            "Effort max |.| (N/m)": "absmax",
            "Rayon du max (m)": "argmax_r",
            "Azimut du max (°)": "argmax_theta",
        }[grandeur]

        extrema = {m: metrics.extremum_table(grid_preds[m], f"{force_col}_pred") for m in (model1, model2)}
        for label, ref_suffix in REF_SOURCES:
            extrema[label] = metrics.extremum_table(castor_df, f"{force_col}_{ref_suffix}")

        col1, col2 = st.columns(2)
        with col1:
            tsr_key = _key(f"tsr_fixe_{force_col}")
            st.session_state.setdefault(tsr_key, tsrs_dispo[len(tsrs_dispo) // 2])
            c1a, c1b = st.columns([3, 1])
            with c1a:
                tsr_fixe = st.selectbox("TSR fixé", tsrs_dispo, key=tsr_key)
            with c1b:
                st.write("")
                st.button("Aléatoire", key=_key(f"tsr_random_{force_col}"), on_click=_pick_random_tsr, args=(tsr_key,))

            series = {}
            for label in [model1, model2] + [l for l, _ in REF_SOURCES]:
                sub = extrema[label][np.isclose(extrema[label]["TSR"], tsr_fixe)].set_index("yaw").reindex(yaws_dispo)
                series[label] = sub[value_col].values
            fig1 = plotting.uniform_curve_figure(yaws_dispo, series, "Yaw (°)", grandeur, f"{grandeur} vs Yaw (TSR={tsr_fixe:g})")
            st.pyplot(fig1)

        with col2:
            yaw_key = _key(f"yaw_fixe_{force_col}")
            st.session_state.setdefault(yaw_key, yaws_dispo[len(yaws_dispo) // 2])
            c2a, c2b = st.columns([3, 1])
            with c2a:
                yaw_fixe = st.selectbox("Yaw fixé (°)", yaws_dispo, key=yaw_key)
            with c2b:
                st.write("")
                st.button("Aléatoire", key=_key(f"yaw_random_{force_col}"), on_click=_pick_random_yaw, args=(yaw_key,))

            series = {}
            for label in [model1, model2] + [l for l, _ in REF_SOURCES]:
                sub = extrema[label][np.isclose(extrema[label]["yaw"], yaw_fixe)].set_index("TSR").reindex(tsrs_dispo)
                series[label] = sub[value_col].values
            fig2 = plotting.uniform_curve_figure(tsrs_dispo, series, "TSR", grandeur, f"{grandeur} vs TSR (yaw={yaw_fixe:g}°)")
            st.pyplot(fig2)

    elif sous_mode in ("Enveloppe C_P", "Enveloppe C_T"):
        coef = "Cp" if sous_mode == "Enveloppe C_P" else "Ct"

        cp_ct = {m: compute_cp(grid_preds[m], "Fn_pred", "Ft_pred") for m in (model1, model2)}
        for label, ref_suffix in REF_SOURCES:
            cp_ct[label] = compute_cp(castor_df, f"Fn_{ref_suffix}", f"Ft_{ref_suffix}")

        def _col(label, ref_suffix=None):
            return f"{coef}_pred" if label in (model1, model2) else f"{coef}_{ref_suffix}"

        col1, col2 = st.columns(2)
        with col1:
            tsr_key = _key(f"tsr_fixe_{coef}")
            st.session_state.setdefault(tsr_key, tsrs_dispo[len(tsrs_dispo) // 2])
            c1a, c1b = st.columns([3, 1])
            with c1a:
                tsr_fixe = st.selectbox("TSR fixé", tsrs_dispo, key=tsr_key)
            with c1b:
                st.write("")
                st.button("Aléatoire", key=_key(f"tsr_random_{coef}"), on_click=_pick_random_tsr, args=(tsr_key,))

            series = {}
            for label in [model1, model2] + [l for l, _ in REF_SOURCES]:
                ref_suffix = dict(REF_SOURCES).get(label)
                sub = cp_ct[label][np.isclose(cp_ct[label]["TSR"], tsr_fixe)].set_index("yaw").reindex(yaws_dispo)
                series[label] = sub[_col(label, ref_suffix)].values
            fig1 = plotting.uniform_curve_figure(yaws_dispo, series, "Yaw (°)", coef, f"{coef} vs Yaw (TSR={tsr_fixe:g})")
            st.pyplot(fig1)

        with col2:
            yaw_key = _key(f"yaw_fixe_{coef}")
            st.session_state.setdefault(yaw_key, yaws_dispo[len(yaws_dispo) // 2])
            c2a, c2b = st.columns([3, 1])
            with c2a:
                yaw_fixe = st.selectbox("Yaw fixé (°)", yaws_dispo, key=yaw_key)
            with c2b:
                st.write("")
                st.button("Aléatoire", key=_key(f"yaw_random_{coef}"), on_click=_pick_random_yaw, args=(yaw_key,))

            series = {}
            for label in [model1, model2] + [l for l, _ in REF_SOURCES]:
                ref_suffix = dict(REF_SOURCES).get(label)
                sub = cp_ct[label][np.isclose(cp_ct[label]["yaw"], yaw_fixe)].set_index("TSR").reindex(tsrs_dispo)
                series[label] = sub[_col(label, ref_suffix)].values
            fig2 = plotting.uniform_curve_figure(tsrs_dispo, series, "TSR", coef, f"{coef} vs TSR (yaw={yaw_fixe:g}°)")
            st.pyplot(fig2)
