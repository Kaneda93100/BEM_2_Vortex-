"""Chargement (mis en cache) des jeux de données utilisés par l'app."""
import json

import pandas as pd
import streamlit as st

from app.core_app.bootstrap import REPO_ROOT

BEM_GRID_CACHE_CSV = REPO_ROOT / "data" / "castor_sven" / "bem_grid_cache.csv"
BEM_GRID_CACHE_META = REPO_ROOT / "data" / "castor_sven" / "bem_grid_cache_meta.json"


@st.cache_data(show_spinner=False)
def load_train_test():
    df_train = pd.read_csv(REPO_ROOT / "data" / "processed" / "train.csv")
    df_test = pd.read_csv(REPO_ROOT / "data" / "processed" / "test.csv")
    return df_train, df_test


@st.cache_data(show_spinner=False)
def load_castor_sven():
    df = pd.read_csv(REPO_ROOT / "data" / "castor_sven" / "forces_BEM_SVEN_CASTOR.csv")
    if "TSR" not in df.columns:
        df["TSR"] = 8.0
    return df


@st.cache_data(show_spinner=False)
def load_bem_grid_cache():
    if not BEM_GRID_CACHE_CSV.exists() or not BEM_GRID_CACHE_META.exists():
        raise FileNotFoundError(
            "Cache BEM introuvable. Lancer d'abord : "
            "python app/scripts/precompute_bem_test_uniforme.py"
        )
    df = pd.read_csv(BEM_GRID_CACHE_CSV)
    with open(BEM_GRID_CACHE_META, encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta
