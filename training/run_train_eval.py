import time
import os
from training.src.data_loader import load_clean_data, get_splits
from training.src.optimize_ae import optimize_and_train_ae
from training.src.optimize import optimize
from training.src.evaluate import evaluator, evaluate_baselines
from training.src.baseline_boost import train_latent_boosting

# Import des constantes globales de configuration
from core.config import TRIALS_GV, TRIALS_GM, TRIALS_AE, LATENT_DIMS_GV, LATENT_DIMS_GM

def format_duration(seconds):
    """Transforme des secondes en un format lisible."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


os.makedirs("training/performance", exist_ok=True)
logs = open("training/performance/logs.txt", 'a')
if logs is None:
    raise ValueError("\nEchec dans l'ouverture de logs.txt.\n")

def main():
    # =========================================================================
    # CONFIGURATION DU TRAVAIL COLLABORATIF (Découpage en 3 Groupes)
    # =========================================================================
    # Ces booléens permettent de lancer uniquement la ou les parties souhaitées.
    
    RUN_GROUP_1 = True  # Groupe 1 : Tous les GV (MLP)
    RUN_GROUP_2 = True  # Groupe 2 : Uniquement GM_f (CNN - Forces)
    RUN_GROUP_3 = True  # Groupe 3 : Uniquement GM_v (CNN - Vitesses)

    global_start = time.time()
    
    print("Chargement des données...")
    df_full = load_clean_data()
    processed_path = os.path.join("data", "processed")
    df_train, df_test = get_splits(df_full, seed=42, test_size=0.2, save_dir=processed_path)
    
    # Évaluation systématique de la baseline BEM pure (avec et sans unité)
    evaluate_baselines(df_test)

    # =========================================================================
    # GROUPE 1 : GV (Deep Learning MLP)
    # =========================================================================
    if RUN_GROUP_1:
        logs.write("\n\n" + "-"*80 + "\n")
        logs.write(f" LOGS DU GROUPE 1 : STRATÉGIES GLOBALES VECTORIELLES (GV) + LightGBM\n")
        logs.write("-" * 80 + "\n\n\n")

        print(f"\n{'#'*80}")
        print(" DÉMARRAGE GROUPE 1 : STRATÉGIES GLOBALES VECTORIELLES (GV) + LightGBM")
        print(f"{'#'*80}")

        # 1.A. Pipeline GV (MLP) 
        gv_experiments = []
        for res_strat in ['0', '1', '2']:
            for inter_strat in ['f', 'v']:
                gv_experiments.append({'entree': 'GV', 'res': res_strat, 'inter': inter_strat, 'trials': TRIALS_GV, 'dims': LATENT_DIMS_GV})
        
        for exp in gv_experiments:
            e, r, i, n_trials = exp['entree'], exp['res'], exp['inter'], exp['trials']
            for dim in exp['dims']:
                suffixe = f"D{dim}" if dim > 0 else "D0"
                model_name = f"{e}_{r}_{i}_{suffixe}"
                model_start = time.time()
                
                msg_header = f"\n\n{'*'*70}\n\n PIPELINE DL (MLP) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}\n\n"
                logs.write(msg_header)
                print(f"\n{'*'*70}\n PIPELINE DL (MLP) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}")

                # Étape 1 : Auto-encodeur
                if dim > 0:
                    logs.write(f"\n   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...\n")
                    print(f"   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...")
                    optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=TRIALS_AE)
                else:
                    logs.write(f"\n   [1/3] Mode D0 : Pas d'Auto-encodeur.\n")
                    print(f"   [1/3] Mode D0 : Pas d'Auto-encodeur.")
                
                # Étape 2 : Optimisation Optuna du réseau de neurones
                logs.write(f"\n   [2/3] Optimisation du Modèle Prédictif...\n")
                print(f"   [2/3] Optimisation du Modèle Prédictif...")
                optimize(df_train, entree=e, residuelle=r, inter=i, suffixe=suffixe, n_trials=n_trials)
                
                # Étape 3 : Entraînement final et Cross-Validation Exhaustive
                logs.write(f"\n   [3/3] Évaluation Finale...\n")
                print(f"   [3/3] Évaluation Finale...")
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, suffixe=suffixe)
                
                msg_end = f"\n--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---\n"
                logs.write(msg_end)
                print(f"--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---")

    # =========================================================================
    # GROUPE 2 : GM (Deep Learning CNN) - INTERMÉDIAIRE 'f'
    # =========================================================================
    if RUN_GROUP_2:
        logs.write("\n\n" + "-"*80 + "\n")
        logs.write(f" LOGS DU GROUPE 2 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Forces (f)\n")
        logs.write("-" * 80 + "\n\n\n")

        print(f"\n{'#'*80}")
        print(" DÉMARRAGE GROUPE 2 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Forces (f)")
        print(f"{'#'*80}")
        
        gm_f_experiments = []
        for res_strat in ['0', '1', '2']:
            gm_f_experiments.append({'entree': 'GM', 'res': res_strat, 'inter': 'f', 'trials': TRIALS_GM, 'dims': LATENT_DIMS_GM})
        
        for exp in gm_f_experiments:
            e, r, i, n_trials = exp['entree'], exp['res'], exp['inter'], exp['trials']
            for dim in exp['dims']:
                suffixe = f"D{dim}" if dim > 0 else "D0"
                model_name = f"{e}_{r}_{i}_{suffixe}"
                model_start = time.time()
                
                logs.write(f"\n\n{'*'*70}\n\n PIPELINE DL (CNN) : {model_name} | Optuna Trials : {n_trials}\n\n{'*'*70}\n")
                print(f"\n{'*'*70}\n PIPELINE DL (CNN) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}")

                if dim > 0:
                    logs.write(f"\n   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...\n")
                    print(f"   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...")
                    optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=TRIALS_AE)
                else:
                    logs.write(f"\n   [1/3] Mode D0 : Pas d'Auto-encodeur.\n")
                    print(f"   [1/3] Mode D0 : Pas d'Auto-encodeur.")
                
                logs.write(f"\n   [2/3] Optimisation du Modèle Prédictif...\n")
                print(f"   [2/3] Optimisation du Modèle Prédictif...")
                optimize(df_train, entree=e, residuelle=r, inter=i, suffixe=suffixe, n_trials=n_trials)
                
                logs.write(f"\n   [3/3] Évaluation Finale...\n")
                print(f"   [3/3] Évaluation Finale...")
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, suffixe=suffixe)
                
                logs.write(f"\n--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---\n")
                print(f"--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---")


    # =========================================================================
    # GROUPE 3 : GM (Deep Learning CNN) - INTERMÉDIAIRE 'v'
    # =========================================================================
    if RUN_GROUP_3:
        logs.write("\n\n" + "-"*80 + "\n")
        logs.write(f" LOGS DU GROUPE 3 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Vitesses (v)\n")
        logs.write("-" * 80 + "\n\n\n")
        
        print(f"\n{'#'*80}")
        print(" DÉMARRAGE GROUPE 3 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Vitesses (v)")
        print(f"{'#'*80}")
        
        gm_v_experiments = []
        for res_strat in ['0', '1', '2']:
            gm_v_experiments.append({'entree': 'GM', 'res': res_strat, 'inter': 'v', 'trials': TRIALS_GM, 'dims': LATENT_DIMS_GM})
        
        for exp in gm_v_experiments:
            e, r, i, n_trials = exp['entree'], exp['res'], exp['inter'], exp['trials']
            for dim in exp['dims']:
                suffixe = f"D{dim}" if dim > 0 else "D0"
                model_name = f"{e}_{r}_{i}_{suffixe}"
                model_start = time.time()
                
                logs.write(f"\n\n{'*'*70}\n PIPELINE DL (CNN) : {model_name} | Optuna Trials : {n_trials}\n\n{'*'*70}\n")
                print(f"\n{'*'*70}\n PIPELINE DL (CNN) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}")

                if dim > 0:
                    logs.write(f"\n   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...\n")  
                    print(f"   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...")
                    optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=TRIALS_AE)
                else:
                    logs.write(f"\n   [1/3] Mode D0 : Pas d'Auto-encodeur.\n")
                    print(f"   [1/3] Mode D0 : Pas d'Auto-encodeur.")
                
                logs.write(f"\n   [2/3] Optimisation du Modèle Prédictif...\n")
                print(f"   [2/3] Optimisation du Modèle Prédictif...")
                optimize(df_train, entree=e, residuelle=r, inter=i, suffixe=suffixe, n_trials=n_trials)
                
                logs.write(f"\n   [3/3] Évaluation Finale...\n")
                print(f"   [3/3] Évaluation Finale...")
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, suffixe=suffixe)
                
                logs.write(f"\n--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---\n")
                print(f"--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---")

  
    for i, run in enumerate([RUN_GROUP_1, RUN_GROUP_2, RUN_GROUP_3], 1):
        if run:
            logs.write(f"\n\nSession terminée (groupe {i}) en {format_duration(time.time() - global_start)}\n")

    logs.close()
    print(f"\nSession globale terminée en {format_duration(time.time() - global_start)}")

if __name__ == "__main__":
    main()