"""Chargement (mis en cache) des jeux de données utilisés par l'app."""
import pandas as pd
import streamlit as st

from app.core_app.bootstrap import REPO_ROOT


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
