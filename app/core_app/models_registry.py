"""Découverte et parsing des modèles entraînés disponibles dans training/models/{GM,GV}."""
import re

from app.core_app.bootstrap import REPO_ROOT

BASELINE_BEM = "baseline_bem"

# {GM|GV}_{résiduelle}_{f|v}_D{ae_nature}{ae_dim}_{A|B}
MODEL_NAME_RE = re.compile(r"^(GM|GV)_(0|1|2\+?)_([fv])_D([A-Z]*)(\d+)_([AB])$")

OPTION_DESCRIPTIONS = {
    "A": "Score A — erreur relative locale : à chaque point (r, theta), "
         "on calcule |prédiction - vérité| / |vérité| (plancher de 1 N/m sur F_t). "
         "Sensible aux zones où la force vraie est faible.",
    "B": "Score B — erreur normalisée par la pression dynamique locale D = 0.5·ρ·V_app²·|corde(r)| : "
         "on calcule |prédiction - vérité| / D. Normalisation physique uniforme sur toute la pale, "
         "moins sensible aux valeurs proches de zéro.",
}


def parse_model_name(model_name: str) -> dict:
    """Parse un nom de modèle (nom de fichier sans extension) en ses composantes."""
    match = MODEL_NAME_RE.match(model_name)
    if not match:
        raise ValueError(f"Nom de modèle non reconnu : '{model_name}'")
    entree, residuelle, inter, ae_nature, ae_dim, option = match.groups()
    return {
        "entree": entree,
        "residuelle": residuelle,
        "inter": inter,
        "ae_nature": ae_nature or None,
        "ae_dim": int(ae_dim),
        "option": option,
    }


def needs_bem(model_name: str) -> bool:
    """True si le modèle a besoin des colonnes BEM en entrée (résiduelle '1', '2' ou '2+')."""
    if model_name == BASELINE_BEM:
        return True
    info = parse_model_name(model_name)
    return info["residuelle"] in ("1", "2", "2+")


def list_available_models() -> list[str]:
    """Liste tous les modèles utilisables : baseline_bem + tous les .pth de training/models/GM|GV."""
    models = [BASELINE_BEM]
    for sub in ("GM", "GV"):
        model_dir = REPO_ROOT / "training" / "models" / sub
        if model_dir.exists():
            for path in sorted(model_dir.glob("*.pth")):
                models.append(path.stem)
    return models


def describe_model(model_name: str) -> str:
    if model_name == BASELINE_BEM:
        return "Baseline BEM (bemol), sans réseau de neurones."
    info = parse_model_name(model_name)
    parts = [f"Entrée {info['entree']}", f"résiduelle {info['residuelle']}", f"intermédiaire {info['inter']}"]
    if info["ae_dim"] > 0:
        parts.append(f"auto-encodeur {info['ae_nature']} (dim {info['ae_dim']})")
    else:
        parts.append("sans auto-encodeur")
    parts.append(f"loss {info['option']}")
    return ", ".join(parts)
