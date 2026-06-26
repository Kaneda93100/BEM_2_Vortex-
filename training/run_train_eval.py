import time
import os
import itertools
from training.src.data_loader import load_clean_data, get_splits
from training.src.optimize_ae import optimize_and_train_ae
from training.src.optimize import optimize
from training.src.evaluate import evaluator, evaluate_baselines
from core.config import INTERMS, AE_NATURES, AE_DIMS, TRIALS_AE, TRIALS_GV, TRIALS_GM

def main():
    os.makedirs("training/performance", exist_ok=True)
    df_full = load_clean_data()
    df_train, df_test = get_splits(df_full, seed=42)
    
    evaluate_baselines(df_test)

    # 1. PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS

    print("\n" + "="*80)
    print(" PHASE 1 : PRÉ-ENTRAÎNEMENT DE LA BANQUE D'AUTO-ENCODEURS")
    print("="*80)
    
    residuelles_ae = ['0', '1', '2']
    
    for r, i, nature, dim in itertools.product(residuelles_ae, INTERMS, AE_NATURES, AE_DIMS):
        t0 = time.perf_counter()
        optimize_and_train_ae(df_train, residuelle=r, inter=i, latent_dim=dim, ae_nature=nature, n_trials=TRIALS_AE)
        print(f"   [CHRONO] AE {r}_{i}_D{nature}{dim} : {time.perf_counter()-t0:.1f}s")

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
        ('GM', '2+', 'f', True,  'A'),
        ('GM', '2+', 'v', True,  'A'),
    ]
    
    for e, r, i, has_ae, opt in test_models:
        

        # On exclut les modes "1+" (qui n'ont pas de sens physique en fait)
        if '1+' in r:
            continue
            
        ae_label = "DXY" if has_ae else "D0"
        model_base_name = f"{e}_{r}_{i}_{ae_label}_{opt}"
        
        print(f"\n\n{'#'*80}")
        print(f" PIPELINE : {model_base_name}")
        print(f"{'#'*80}")
        
        n_trials_current = TRIALS_GV if e == 'GV' else TRIALS_GM
        
        t0 = time.perf_counter()
        optimize(df_train, entree=e, residuelle=r, inter=i, has_ae=has_ae, option=opt, model_base_name=model_base_name, n_trials=n_trials_current)
        print(f"   [CHRONO] optimize {model_base_name} : {time.perf_counter()-t0:.1f}s")

        t0 = time.perf_counter()
        evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, has_ae=has_ae, option=opt)
        print(f"   [CHRONO] evaluator {model_base_name} : {time.perf_counter()-t0:.1f}s")

if __name__ == "__main__":
    main()