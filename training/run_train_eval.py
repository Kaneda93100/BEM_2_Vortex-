import time
import os
import pandas as pd
import itertools
from training.src.data_loader import load_clean_data, get_splits, subsample_train
from training.src.optimize_ae import optimize_and_train_ae
from training.src.optimize import optimize
from training.src.evaluate import evaluator, evaluate_baselines
from core.config import (INTERMS, AE_NATURES, AE_DIMS, TRIALS_AE, TRIALS_GV, TRIALS_GM, DATA_PCTS,
                          DEFAULT_BEM_SUFFIX, needs_bem_suffix, get_ae_residual_key,
                          format_model_name)

session = 'S2'

if session == 'S1' :
    print('LANCEMENT DE LA SESSION 1\n\n')
    test_models = [
        ('GM', '0', 'v', True, 'A'), # Score A (P50) : 34.8 (Bon score) 
        ('GV', '0', 'f', True, 'A') # Score A (P50) : 29.5 (Meilleur score)
    ]
elif session == 'S2' :
    print('LANCEMENT DE LA SESSION 2\n\n')
    test_models = [
        ('GM', '1', 'f', False, 'A'), # Score A (P50) : 29.9 (Très bon score)
        ('GV', '2', 'v', True, 'A')  # Score A (P50) : 37.0 (Score moyen)
    ]
elif session == 'S3' :
    print('LANCEMENT DE LA SESSION 3\n\n')
    test_models =  [
        ('GV', '2', 'v', False, 'B'), # Score B (P50) : 3.27 (Meilleur score)
        ('GV', '0', 'v', True, 'B') # Score B (P50) : 3.64 (Deuxième meilleur)
    ]
elif session == 'S4' : 
    print('LANCEMENT DE LA SESSION 4\n\n')
    test_models = [
        ('GM', '1', 'f', False, 'B'),  # Score B : 3.49 (Bon score) 
        ('GM', '0', 'v', False, 'B'), # Score B : 6.16 (Mauvais score)
    ]


def main():
    os.makedirs("training/performance", exist_ok=True)
    df_full = load_clean_data()
    df_train, df_test = get_splits(df_full, seed=42)

    baseline_scores = evaluate_baselines(df_test, DEFAULT_BEM_SUFFIX)
    # 1. PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS

    print("\n" + "="*80)
    print(" PHASE 1 : PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS")
    print("="*80)

    residuelles_ae = ['0', '1']

    for r, i, nature, dim in itertools.product(residuelles_ae, INTERMS, AE_NATURES, AE_DIMS):
            # Seule l'AE '1' dépend de la variante BEM (cible Y = SVEN - BEM) ; '0' reste unique.
            bem_suffix = DEFAULT_BEM_SUFFIX if get_ae_residual_key(r) == '1' else None
            t0 = time.perf_counter()
            optimize_and_train_ae(df_train, residuelle=r, inter=i, latent_dim=dim, ae_nature=nature, n_trials=TRIALS_AE, bem_suffix=bem_suffix)
            tag = f"{r}_{i}_D{nature}{dim}" + (f"_{bem_suffix}" if bem_suffix else "")
            print(f"   [CHRONO] AE {tag} : {time.perf_counter()-t0:.1f}s")

    # 2. PLAN D'EXPÉRIENCES (13 MODÈLES DE LA SESSION)

    print("\n" + "="*80)
    print(f" PHASE 2 : PLAN D'EXPÉRIENCES")
    print("="*80)

    for pct in DATA_PCTS:
            df_train_pct = subsample_train(df_train, pct)

            print("\n" + "="*80)
            print(f" PROPORTION DE DONNÉES D'ENTRAÎNEMENT : {pct}% "
                f"({df_train_pct['yaw'].nunique()}/{df_train['yaw'].nunique()} yaw)")
            print("="*80)

            for e, r, i, has_ae, opt in test_models:

                ae_label = "DXY" if has_ae else "D0"
                bem_suffix = DEFAULT_BEM_SUFFIX if needs_bem_suffix(r) else None
                model_base_name = format_model_name(e, r, i, ae_label, opt, pct, bem_suffix)

                print(f"\n\n{'#'*80}")
                print(f" PIPELINE : {model_base_name}")
                print(f"{'#'*80}")

                n_trials_current = TRIALS_GV if e == 'GV' else TRIALS_GM

                t0 = time.perf_counter()
                optimize(df_train_pct, entree=e, residuelle=r, inter=i, has_ae=has_ae, option=opt, model_base_name=model_base_name, n_trials=n_trials_current, bem_suffix=bem_suffix)
                print(f"   [CHRONO] optimize {model_base_name} : {time.perf_counter()-t0:.1f}s")

                t0 = time.perf_counter()
                evaluator(df_train_pct, df_test, entree=e, residuelle=r, inter=i, has_ae=has_ae, option=opt, baseline_scores=baseline_scores, pct=pct, bem_suffix=bem_suffix)
                print(f"   [CHRONO] evaluator {model_base_name} : {time.perf_counter()-t0:.1f}s")

    

if __name__ == "__main__":
    main()
