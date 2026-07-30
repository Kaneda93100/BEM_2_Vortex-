import time
import os
import itertools
from training.src.data_loader import load_clean_data, get_splits, subsample_train
from training.src.optimize_ae import optimize_and_train_ae
from training.src.optimize import optimize
from training.src.evaluate import evaluator, evaluate_baselines
from core.config import (INTERMS, AE_NATURES, AE_DIMS, TRIALS_AE, TRIALS_GV, TRIALS_GM, DATA_PCTS,
                          BEM_SUFFIXES, DEFAULT_BEM_SUFFIX, needs_bem_suffix, get_ae_residual_key,
                          format_model_name)

def main():
    os.makedirs("training/performance", exist_ok=True)
    df_full = load_clean_data()
    df_train, df_test = get_splits(df_full, seed=42)

    baseline_scores_by_suffix = {suf: evaluate_baselines(df_test, suf) for suf in BEM_SUFFIXES}

    # 1. PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS

    print("\n" + "="*80)
    print(" PHASE 1 : PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS")
    print("="*80)
    
    # '2' est omis : il produit la même cible Y (SVEN) que '0', donc le même auto-encodeur
    # (voir core.config.get_ae_residual_key, utilisé par optimize_and_train_ae pour dédupliquer).
    residuelles_ae = ['0', '1']

    for r, i, nature, dim in itertools.product(residuelles_ae, INTERMS, AE_NATURES, AE_DIMS):
        # Seule l'AE '1' dépend de la variante BEM (cible Y = SVEN - BEM) ; '0' reste unique.
        suffixes_for_ae = BEM_SUFFIXES if get_ae_residual_key(r) == '1' else [None]
        for bem_suffix in suffixes_for_ae:
            t0 = time.perf_counter()
            optimize_and_train_ae(df_train, residuelle=r, inter=i, latent_dim=dim, ae_nature=nature, n_trials=TRIALS_AE, bem_suffix=bem_suffix)
            tag = f"{r}_{i}_D{nature}{dim}" + (f"_{bem_suffix}" if bem_suffix else "")
            print(f"   [CHRONO] AE {tag} : {time.perf_counter()-t0:.1f}s")

      # 2. PLAN D'EXPÉRIENCES (14 MODÈLES CIBLÉS)

    print("\n" + "="*80)
    print(" PHASE 2 : PLAN D'EXPÉRIENCES (14 MODÈLES STRATÉGIQUES)")
    print("="*80)
    
    test_models = [
        # Q1: Comparaison de l'intégration BEM (0 vs 1 vs 2)
        ('GM', '0', 'f', False, 'A'),
        ('GM', '1', 'f', False, 'A'), # Point de Pivot
        ('GM', '2', 'f', False, 'A'),
        
        # Q2: Impact du format spatial (GM vs GV)
        ('GV', '1', 'f', False, 'A'),
        
        # Q3: Grandeur cible (Forces vs Vitesses)
        ('GM', '1', 'v', False, 'A'),
        
        # Q4: Fonction de perte (A vs B)
        ('GM', '1', 'f', False, 'B'),
        ('GM', '1', 'v', False, 'B'),
        
        # Q5: Apport de l'Auto-Encodeur (D0 vs DXY)
        ('GM', '0', 'f', True,  'A'),
        ('GM', '1', 'f', True,  'A'),
        ('GM', '2', 'f', True,  'A'), # Servira de base pour comparer le 2+
        ('GM', '1', 'v', True,  'A'),
        ('GM', '2', 'v', True,  'A'), # Servira de base pour comparer le 2+
        
        # Q6: Apport du mode '+' (Projection dans l'espace latent)
        ('GV', '2+', 'f', True,  'A'),
        ('GV', '2+', 'v', True,  'A'),
    ]
    
    for pct in DATA_PCTS:
        df_train_pct = subsample_train(df_train, pct)

        print("\n" + "="*80)
        print(f" PROPORTION DE DONNÉES D'ENTRAÎNEMENT : {pct}% "
              f"({df_train_pct['yaw'].nunique()}/{df_train['yaw'].nunique()} yaw)")
        print("="*80)

        for e, r, i, has_ae, opt in test_models:

            # On exclut les modes "1+" (qui n'ont pas de sens physique en fait)
            if '1+' in r:
                continue

            ae_label = "DXY" if has_ae else "D0"
            suffixes_for_model = BEM_SUFFIXES if needs_bem_suffix(r) else [None]

            for bem_suffix in suffixes_for_model:
                model_base_name = format_model_name(e, r, i, ae_label, opt, pct, bem_suffix)
                baseline_scores = baseline_scores_by_suffix[bem_suffix or DEFAULT_BEM_SUFFIX]

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