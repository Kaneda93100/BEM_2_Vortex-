import time
import os
from src.data_loader import load_clean_data, get_splits
from src.optimize import optimize
from src.evaluate import evaluator, evaluate_baselines

def format_duration(seconds):
    """Transforme des secondes en un format lisible."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"

def main():
    # Début du chrono global
    global_start = time.time()
    
    # 1. Chargement des données
    print("Chargement des données...")
    df_full = load_clean_data()
   
    processed_path = os.path.join("data", "processed")
    # Split unique par Yaw pour toutes les stratégies
    df_train, df_test = get_splits(df_full, seed=42, test_size=0.3, save_dir=processed_path)
    
    # 2. Initialisation de la Baseline 
    evaluate_baselines(df_test)

    # 3. Les stratégies à tester
    entrees = ['G','P']
    residuelles = ['0', '1']
    inters = ['f', 'v']
    
    # 4. Boucles d'expérimentation
    for e in entrees:
        print(f"\n" + "#"*40)
        print(f" STRATÉGIE SPATIALE : {e}")
        print("#"*40)
        
        for r in residuelles:
            for i in inters:
                model_start = time.time()
                model_name = f"{e}_{r}_{i}"
                
                print(f"\n=== Modèle : {model_name} ===")
                
                # --- PHASE 1 : OPTIMISATION ---
     
                print(f"   [1/2] Optimisation Optuna en cours...")
                opt_start = time.time()
                optimize(df_train, entree=e, residuelle=r, inter=i, n_trials=2)
                opt_duration = time.time() - opt_start
                print(f"   >> Temps Optimisation : {format_duration(opt_duration)}")
                
                # --- PHASE 2 : ÉVALUATION ---
                print(f"   [2/2] Ré-entraînement final et évaluation...")
                eval_start = time.time()
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i)
                eval_duration = time.time() - eval_start
                print(f"   >> Temps Évaluation : {format_duration(eval_duration)}")
                
                # Temps total pour ce modèle précis
                model_total = time.time() - model_start
                print(f"--- Modèle {model_name} terminé en {format_duration(model_total)} ---")

    # 5. Synthèse finale
    print(f"\n" + "="*50)
    total_campaign_duration = time.time() - global_start
    print(f"CAMPAGNE TERMINÉE !")
    print(f"Temps total d'exécution : {format_duration(total_campaign_duration)}")
    print("="*50)

if __name__ == "__main__":
    main()