"""Script PROVISOIRE de vérification : compare la BEM recalculée via bemol (app/core_app/
bem_provider.py) aux colonnes Fn_BEM/Ft_BEM/V_eff_BEM/alpha_BEM déjà présentes dans
data/processed/train.csv et test.csv.
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from app.core_app.bem_provider import compute_bem


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _report(df_ref, df_bem, label):
    merged = df_ref.sort_values(["yaw", "TSR", "r", "theta"], kind="mergesort").reset_index(drop=True)
    bem = df_bem.sort_values(["yaw", "TSR", "r", "theta"], kind="mergesort").reset_index(drop=True)
    if len(merged) != len(bem):
        print(f"[{label}] ECHEC : desalignement ({len(merged)} vs {len(bem)} lignes).")
        return

    print(f"\n=== {label} ({len(merged)} lignes, {merged[['yaw', 'TSR']].drop_duplicates().shape[0]} couples) ===")
    for col_ref, col_bem in [
        ("Fn_BEM", "Fn_BEM"), ("Ft_BEM", "Ft_BEM"),
        ("V_eff_BEM", "V_eff_BEM"), ("alpha_BEM", "alpha_BEM"),
    ]:
        ref_vals, bem_vals = merged[col_ref].values, bem[col_bem].values
        rmse = _rmse(ref_vals, bem_vals)
        max_abs = float(np.max(np.abs(ref_vals - bem_vals)))
        mean_abs_ref = float(np.mean(np.abs(ref_vals)))
        rel_pct = rmse / mean_abs_ref * 100 if mean_abs_ref > 0 else float("nan")
        print(f"  {col_ref:12s} RMSE={rmse:10.4f}  max|ecart|={max_abs:10.4f}  RMSE relatif={rel_pct:6.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=None, help="Limiter le nombre de couples (yaw,TSR) testés (défaut : tous).")
    args = parser.parse_args()

    df_train = pd.read_csv(_REPO_ROOT / "data" / "processed" / "train.csv")
    df_test = pd.read_csv(_REPO_ROOT / "data" / "processed" / "test.csv")

    for label, df_ref in [("train.csv", df_train), ("test.csv", df_test)]:
        pairs = df_ref[["yaw", "TSR"]].drop_duplicates()
        if args.n_pairs:
            pairs = pairs.head(args.n_pairs)
        pairs_list = list(pairs.itertuples(index=False, name=None))
        df_ref_subset = df_ref.merge(pairs, on=["yaw", "TSR"])

        print(f"\nCalcul bemol pour {len(pairs_list)} couple(s) (yaw, TSR) de {label}…")
        df_bem = compute_bem(pairs_list, nbr_az=72)
        _report(df_ref_subset, df_bem, label)


if __name__ == "__main__":
    main()
