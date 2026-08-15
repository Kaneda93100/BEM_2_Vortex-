"""Découverte et parsing des modèles entraînés disponibles dans training/models/{GM,GV}."""
import re

from app.core_app.bootstrap import REPO_ROOT
from core.config import BEM_SUFFIXES, DEFAULT_BEM_SUFFIX, needs_bem_suffix

# {GM|GV}_{résiduelle}_{f|v}[_{suffixe BEM}]_D{ae_nature}{ae_dim}_{A|B}_P{pourcentage_donnees_train}
# Le segment suffixe BEM (DUM/IFP/P&P) n'est présent que lorsque needs_bem_suffix(résiduelle) est vrai.
_BEM_SUFFIX_ALT = "|".join(re.escape(s) for s in BEM_SUFFIXES)
MODEL_NAME_RE = re.compile(
    rf"^(GM|GV)_(0|1|2\+?)_([fv])(?:_({_BEM_SUFFIX_ALT}))?_D([A-Z]*)(\d+)_([AB])_P(\d{{1,3}})$"
)

# baseline_bem_{suffixe} : prédiction BEM brute (bemol), sans réseau de neurones, une entrée par
# variante de correction. Compatible avec les scores A et B (pas d'option d'entraînement propre).
BASELINE_PREFIX = "baseline_bem"
BASELINE_MODELS = {f"{BASELINE_PREFIX}_{suffix}": suffix for suffix in BEM_SUFFIXES}
BASELINE_BEM = f"{BASELINE_PREFIX}_{DEFAULT_BEM_SUFFIX}"

OPTION_DESCRIPTIONS = {
    "A": "Score A — erreur relative locale : à chaque point (r, theta), "
         "on calcule |prédiction - vérité| / |vérité| (plancher de 1 N/m sur F_t). "
         "Sensible aux zones où la force vraie est faible.",
    "B": "Score B — erreur normalisée par la pression dynamique locale D = 0.5·ρ·V_app²·|corde(r)| : "
         "on calcule |prédiction - vérité| / D. Normalisation physique uniforme sur toute la pale, "
         "moins sensible aux valeurs proches de zéro.",
}


def is_baseline(model_name: str) -> bool:
    return model_name in BASELINE_MODELS


def parse_model_name(model_name: str) -> dict:
    """Parse un nom de modèle (nom de fichier sans extension) en ses composantes."""
    match = MODEL_NAME_RE.match(model_name)
    if not match:
        raise ValueError(f"Nom de modèle non reconnu : '{model_name}'")
    entree, residuelle, inter, bem_suffix, ae_nature, ae_dim, option, pct = match.groups()
    needs_suffix = needs_bem_suffix(residuelle)
    if needs_suffix and bem_suffix is None:
        raise ValueError(f"Nom de modèle '{model_name}' : suffixe BEM manquant pour résiduelle={residuelle!r}.")
    if not needs_suffix and bem_suffix is not None:
        raise ValueError(f"Nom de modèle '{model_name}' : suffixe BEM inattendu pour résiduelle={residuelle!r}.")
    return {
        "entree": entree,
        "residuelle": residuelle,
        "inter": inter,
        "bem_suffix": bem_suffix,
        "ae_nature": ae_nature or None,
        "ae_dim": int(ae_dim),
        "option": option,
        "pct": int(pct),
    }


def model_bem_suffix(model_name: str) -> str | None:
    """Suffixe BEM (DUM/IFP/P&P) requis en entrée par ce modèle, ou None s'il n'en a besoin d'aucun."""
    if is_baseline(model_name):
        return BASELINE_MODELS[model_name]
    return parse_model_name(model_name)["bem_suffix"]


def model_option(model_name: str) -> str | None:
    """Option d'entraînement ('A'/'B') du modèle, ou None pour un baseline (compatible avec les deux)."""
    if is_baseline(model_name):
        return None
    return parse_model_name(model_name)["option"]


def needs_bem(model_name: str) -> bool:
    """True si le modèle a besoin des colonnes BEM en entrée (baseline, ou résiduelle '1', '2' ou '2+')."""
    return model_bem_suffix(model_name) is not None


def list_available_models() -> list[str]:
    """Liste tous les modèles utilisables : baselines BEM (une par suffixe) + tous les .pth de
    training/models/GM|GV."""
    models = list(BASELINE_MODELS.keys())
    for sub in ("GM", "GV"):
        model_dir = REPO_ROOT / "training" / "models" / sub
        if model_dir.exists():
            for path in sorted(model_dir.glob("*.pth")):
                models.append(path.stem)
    return models


def models_for_option(option: str) -> list[str]:
    """Sous-ensemble de list_available_models() compatible avec le score `option` ('A' ou 'B') :
    les baselines BEM (toujours compatibles) + les modèles entraînés avec cette option."""
    return [m for m in list_available_models() if model_option(m) in (None, option)]


def models_using_bem(option: str) -> list[str]:
    """Sous-ensemble de models_for_option(option) restreint aux modèles qui utilisent réellement
    la BEM en entrée avec une résiduelle '1', '2' ou '2+' (donc sensibles à une perturbation de
    cette entrée) — exclut les baselines BEM brutes (pas de réseau de neurones à sonder) et les
    modèles résiduelle '0' (n'utilisent pas la BEM en entrée)."""
    return [
        m for m in models_for_option(option)
        if not is_baseline(m) and needs_bem_suffix(parse_model_name(m)["residuelle"])
    ]


def describe_model(model_name: str) -> str:
    if is_baseline(model_name):
        return f"Baseline BEM (bemol, correction {BASELINE_MODELS[model_name]}), sans réseau de neurones."
    info = parse_model_name(model_name)
    parts = [f"Entrée {info['entree']}", f"résiduelle {info['residuelle']}", f"intermédiaire {info['inter']}"]
    if info["bem_suffix"]:
        parts.append(f"BEM {info['bem_suffix']}")
    if info["ae_dim"] > 0:
        parts.append(f"auto-encodeur {info['ae_nature']} (dim {info['ae_dim']})")
    else:
        parts.append("sans auto-encodeur")
    parts.append(f"loss {info['option']}")
    parts.append(f"{info['pct']}% des données (yaw,TSR) d'entraînement")
    return ", ".join(parts)
