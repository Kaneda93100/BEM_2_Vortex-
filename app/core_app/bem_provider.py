"""On utilise bemol.DataMaker.src.SimEnv pour calculer la BEM (Fn, Ft, V_eff, alpha)
pour une liste de couples (yaw, TSR), sur la grille native des modèles (36 rayons x 72 azimuts).

On appelle directement la classe SimEnv (bemol/DataMaker/src/SimEnv.py) et sa méthode
data_maker

Trois variantes de correction de sillage en yaw sont disponibles : DUM (aucune correction, 
bemol.bemol.secondary.yawModel.Dummy), IFP (IFPEN) et P&P (Pitt & Peters).
"""
import importlib.util
import sys

import pandas as pd

from app.core_app.bootstrap import REPO_ROOT
from core.config import DEFAULT_BEM_SUFFIX

_BEMOL_PKG_PARENT = REPO_ROOT / "bemol"
if str(_BEMOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_BEMOL_PKG_PARENT))

_SIMENV_PATH = REPO_ROOT / "bemol" / "DataMaker" / "src" / "SimEnv.py"

_simenv_module = None
_envs: dict[str, object] = {}

_YAW_MODEL_CLASS_NAMES = {"DUM": "Dummy", "IFP": "IFPEN", "P&P": "PittAndPeters"}


def _get_simenv_module():
    global _simenv_module
    if _simenv_module is None:
        spec = importlib.util.spec_from_file_location("bemol_simenv", str(_SIMENV_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _simenv_module = module
    return _simenv_module


def _corrections_for_suffix(simenv_mod, bem_suffix: str) -> list:
    """Reproduit la liste de corrections par défaut de SimEnv.py (Prandtl/Burton/Buhl + modèle de
    yaw), en ne changeant que le modèle de yaw selon `bem_suffix`."""
    if bem_suffix not in _YAW_MODEL_CLASS_NAMES:
        raise ValueError(f"Suffixe BEM inconnu : '{bem_suffix}' (attendu parmi {list(_YAW_MODEL_CLASS_NAMES)}).")
    bem = simenv_mod.bem
    yaw_cls = getattr(bem.secondary.yawModel, _YAW_MODEL_CLASS_NAMES[bem_suffix])
    return [
        bem.secondary.hubTipLoss.Prandtl,
        bem.secondary.skewAngle.Burton,
        bem.secondary.turbulentWakeState.Buhl,
        yaw_cls,
    ]


def _get_env(bem_suffix: str):
    """Instance SimEnv réutilisée par suffixe BEM"""
    if bem_suffix not in _envs:
        simenv_mod = _get_simenv_module()
        corrections = _corrections_for_suffix(simenv_mod, bem_suffix)
        _envs[bem_suffix] = simenv_mod.SimEnv(
            omega=simenv_mod.rotor_vortex.omegaRated, U=simenv_mod.rotor_vortex.windRated,
            yaw=0.0, skew=0.0, corrections=corrections,
        )
    return _envs[bem_suffix]


def compute_bem(pairs, bem_suffix: str = DEFAULT_BEM_SUFFIX, nbr_az: int = 72, progress_cb=None) -> pd.DataFrame:
    """Calcule Fn_BEM, Ft_BEM, V_eff_BEM, alpha_BEM pour chaque couple (yaw_deg, tsr) de `pairs`,
    avec la variante de correction `bem_suffix` (DUM/IFP/P&P), sur la grille complète via bemol.DataMaker.src.SimEnv.SimEnv.data_maker.
    Retourne un DataFrame trié par (yaw, TSR, r, theta), colonnes :
    yaw, TSR, r, theta, Fn_BEM, Ft_BEM, V_eff_BEM, alpha_BEM.
    """
    env = _get_env(bem_suffix)
    pairs = [(float(y), float(t)) for y, t in pairs]
    total = len(pairs)

    frames = []
    for i, (yaw, tsr) in enumerate(pairs):
        frames.append(env.data_maker([yaw], [tsr], nbr_az, export=False))
        if progress_cb is not None:
            progress_cb(i + 1, total)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["yaw", "TSR", "r", "theta", "Fn", "Ft", "V_eff", "Alpha_deg"]
    )
    df = df.rename(columns={"Fn": "Fn_BEM", "Ft": "Ft_BEM", "V_eff": "V_eff_BEM", "Alpha_deg": "alpha_BEM"})
    # data_maker fait np.degrees(np.radians(yaw)) en interne, ce qui introduit une erreur
    # d'arrondi flottant (~1e-14) sur la colonne 'yaw' (ex: 15.0 -> 14.999999999999998). Sans
    # incidence sur les calculs physiques, mais on préfère arrondir.
    df["yaw"] = df["yaw"].round(9)
    return df.sort_values(["yaw", "TSR", "r", "theta"], kind="mergesort").reset_index(drop=True)


def compute_bem_multi(pairs, bem_suffixes, nbr_az: int = 72, progress_cb=None) -> pd.DataFrame:
    """Comme compute_bem, mais calcule plusieurs variantes BEM (`bem_suffixes`, ex. BEM_SUFFIXES) et
    les fusionne en colonnes suffixées Fn_BEM_{suffixe}/Ft_BEM_{suffixe}/V_eff_BEM_{suffixe}/
    alpha_BEM_{suffixe} sur une même grille (yaw, TSR, r, theta).
    """
    pairs = list(pairs)
    total = len(pairs) * len(bem_suffixes)
    merged = None
    done_before = 0
    for bem_suffix in bem_suffixes:
        def _cb(done, _n, _base=done_before):
            if progress_cb is not None:
                progress_cb(_base + done, total)

        df_suffix = compute_bem(pairs, bem_suffix=bem_suffix, nbr_az=nbr_az, progress_cb=_cb)
        df_suffix = df_suffix.rename(columns={
            "Fn_BEM": f"Fn_BEM_{bem_suffix}",
            "Ft_BEM": f"Ft_BEM_{bem_suffix}",
            "V_eff_BEM": f"V_eff_BEM_{bem_suffix}",
            "alpha_BEM": f"alpha_BEM_{bem_suffix}",
        })
        merged = df_suffix if merged is None else merged.merge(df_suffix, on=["yaw", "TSR", "r", "theta"])
        done_before += len(pairs)
    return merged


def attach_bem_columns(df: pd.DataFrame, bem_suffix: str = DEFAULT_BEM_SUFFIX, nbr_az: int = 72, progress_cb=None) -> pd.DataFrame:
    """Recalcule via bemol (variante `bem_suffix`) les colonnes Fn_BEM/Ft_BEM/V_eff_BEM/alpha_BEM de
    `df` (qui doit contenir yaw, TSR, r, theta) et les remplace. `df` est réaligné (trié par
    yaw,TSR,r,theta)."""
    pairs = list(df[["yaw", "TSR"]].drop_duplicates().itertuples(index=False, name=None))
    bem_df = compute_bem(pairs, bem_suffix=bem_suffix, nbr_az=nbr_az, progress_cb=progress_cb)
    df_sorted = df.sort_values(["yaw", "TSR", "r", "theta"], kind="mergesort").reset_index(drop=True)
    if len(df_sorted) != len(bem_df):
        raise ValueError(
            f"Désalignement grille BEM : {len(df_sorted)} lignes attendues, {len(bem_df)} calculées par bemol."
        )
    for col in ("Fn_BEM", "Ft_BEM", "V_eff_BEM", "alpha_BEM"):
        df_sorted[col] = bem_df[col].values
    return df_sorted
