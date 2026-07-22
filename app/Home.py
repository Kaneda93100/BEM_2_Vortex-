import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="BEM 2 Vortex", layout="wide")

st.title("BEM 2 Vortex — Exploration des modèles de substitution")

st.markdown(
    """
Cette application permet d'explorer et de comparer les modèles de substitution (réseaux de
neurones GM/GV) entraînés pour prédire les efforts aérodynamiques d'une éolienne en yaw, face au
calcul BEM classique (via bemol) et aux données de référence SVEN et Castor (vortex).

Utilisez la barre latérale pour naviguer entre les modules :
"""
)

st.subheader("Test LHS")
st.markdown(
    """
Compare deux modèles au choix sur les points échantillonnés en LHS de
`train.csv` et `test.csv` (80 + 20 couples (yaw, TSR), grille 36×72 (r, theta)).

- **Paramètres** : choix des deux modèles à comparer et de la métrique (Score A ou B), calcul des
  prédictions sur train et test.
- **Carte F_n / Carte F_t** : carte polaire de l'erreur pour un couple (yaw, TSR) du jeu de test.
- **Enveloppe C_P / Enveloppe C_T** : nuage de points (yaw, TSR) coloré par l'erreur sur le
  coefficient de puissance ou de poussée.
"""
)

st.subheader("Test Uniforme")
st.markdown(
    """
Même structure que Test LHS, mais basé sur `data/castor_sven/forces_BEM_SVEN_CASTOR.csv`
(actuellement TSR=8 fixe, yaw de 5° à 30°). Au lieu de tracer des erreurs, cette page trace
directement les grandeurs physiques (BEM baseline, modèle choisi, SVEN, Castor) en fonction du
yaw, sous forme de courbes continues.
"""
)

st.subheader("Générateur")
st.markdown(
    """
Génère un fichier CSV téléchargeable de prédictions (F_n, F_t, + V_eff, alpha pour les modèles à
intermédiaire vitesse) pour un modèle choisi, soit pour un point (yaw, TSR) unique, soit pour une
grille uniforme de (yaw, TSR) définie par plages et pas de discrétisation.
"""
)
