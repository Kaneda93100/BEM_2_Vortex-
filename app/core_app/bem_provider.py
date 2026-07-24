"""Wrapper autour de bemol.DataMaker.src.SimEnv pour calculer la BEM (Fn, Ft, V_eff, alpha)
pour une liste de couples (yaw, TSR), sur la grille native des modèles (36 rayons x 72 azimuts).

On appelle directement la classe SimEnv (bemol/DataMaker/src/SimEnv.py) et sa méthode
data_maker — le code source réel utilisé pour générer les données du projet —

"""
import importlib.util
import sys

import pandas as pd

from app.core_app.bootstrap import REPO_ROOT

_BEMOL_PKG_PARENT = REPO_ROOT / "bemol"
if str(_BEMOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_BEMOL_PKG_PARENT))

_SIMENV_PATH = REPO_ROOT / "bemol" / "DataMaker" / "src" / "SimEnv.py"

_simenv_module = None
_env = None


def _get_simenv_module():
    global _simenv_module
    if _simenv_module is None:
        spec = importlib.util.spec_from_file_location("bemol_simenv", str(_SIMENV_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _simenv_module = module
    return _simenv_module


def _get_env():
    """Instance SimEnv réutilisée; si j'ai bien compris : 
    omega/U/yaw/skew du constructeur ne servent qu'à des valeurs d'affichage internes (self.tsr) ;
    data_maker recalcule tout par couple (yaw, TSR) via set_wind()/np.radians(yaws[i])."""
    global _env
    if _env is None:
        simenv_mod = _get_simenv_module()
        _env = simenv_mod.SimEnv(omega=simenv_mod.rotor_vortex.omegaRated, U=simenv_mod.rotor_vortex.windRated, yaw=0.0, skew=0.0)
    return _env


def compute_bem(pairs, nbr_az: int = 72) -> pd.DataFrame:
    """Calcule Fn_BEM, Ft_BEM, V_eff_BEM, alpha_BEM pour chaque couple (yaw_deg, tsr) de `pairs`,
    sur la grille complète (rayons de la pale x azimuts uniformément répartis sur [0, 360[),
    via bemol.DataMaker.src.SimEnv.SimEnv.data_maker
    Retourne un DataFrame trié par (yaw, TSR, r, theta), colonnes :
    yaw, TSR, r, theta, Fn_BEM, Ft_BEM, V_eff_BEM, alpha_BEM.
    """
    env = _get_env()
    yaws = [float(y) for y, _ in pairs]
    tsrs = [float(t) for _, t in pairs]
    df = env.data_maker(yaws, tsrs, nbr_az, export=False)
    df = df.rename(columns={"Fn": "Fn_BEM", "Ft": "Ft_BEM", "V_eff": "V_eff_BEM", "Alpha_deg": "alpha_BEM"})
    # data_maker fait np.degrees(np.radians(yaw)) en interne, ce qui introduit une erreur
    # d'arrondi flottant (~1e-14) sur la colonne 'yaw' (ex: 15.0 -> 14.999999999999998). Sans
    # incidence sur les calculs physiques, mais ça casse tout rapprochement exact (reindex/merge)
    # avec des yaws de référence (ex: page Test Uniforme vs data/castor_sven). On arrondit donc
    # 'yaw' pour restaurer l'égalité stricte avec les valeurs demandées en entrée.
    df["yaw"] = df["yaw"].round(9)
    return df.sort_values(["yaw", "TSR", "r", "theta"], kind="mergesort").reset_index(drop=True)


def attach_bem_columns(df: pd.DataFrame, nbr_az: int = 72) -> pd.DataFrame:
    """Recalcule via bemol les colonnes Fn_BEM/Ft_BEM/V_eff_BEM/alpha_BEM de `df` (qui doit
    contenir yaw, TSR, r, theta) et les remplace. `df` est réaligné (trié par yaw,TSR,r,theta). """
    pairs = list(df[["yaw", "TSR"]].drop_duplicates().itertuples(index=False, name=None))
    bem_df = compute_bem(pairs, nbr_az=nbr_az)
    df_sorted = df.sort_values(["yaw", "TSR", "r", "theta"], kind="mergesort").reset_index(drop=True)
    if len(df_sorted) != len(bem_df):
        raise ValueError(
            f"Désalignement grille BEM : {len(df_sorted)} lignes attendues, {len(bem_df)} calculées par bemol."
        )
    for col in ("Fn_BEM", "Ft_BEM", "V_eff_BEM", "alpha_BEM"):
        df_sorted[col] = bem_df[col].values
    return df_sorted
