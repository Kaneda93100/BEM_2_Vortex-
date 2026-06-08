import time
import os
from src.data_loader import load_clean_data, get_splits
from src.optimize_ae import optimize_and_train_ae
from src.optimize import optimize
from src.evaluate import evaluator, evaluate_baselines
from src.baseline_boost import train_latent_boosting

def format_duration(seconds):
    """Transforme des secondes en un format lisible."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"

logs = open("performance/logs.txt", 'a')
if logs == None :
    raise ValueError("\nEchec dans l'ouverture de logs.txt.\n")

def main():
    # =========================================================================
    # CONFIGURATION DU TRAVAIL COLLABORATIF (Découpage en 3 Groupes)
    # =========================================================================
    # Il faut modifier ces booléens pour lancer uniquement sa partie.
    
    RUN_GROUP_1 = True   # Groupe 1 : Tous les GV (MLP) + LightGBM
    RUN_GROUP_2 = True  # Groupe 2 : Uniquement GM_1_f (CNN)
    RUN_GROUP_3 = True  # Groupe 3 : Uniquement GM_1_v (CNN)

    global_start = time.time()
    
    print("Chargement des données...")
    df_full = load_clean_data()
    processed_path = os.path.join("data", "processed")
    df_train, df_test = get_splits(df_full, seed=42, test_size=0.2, save_dir=processed_path)
    
    # Évaluation systématique de la baseline BEM pure
    evaluate_baselines(df_test)

    # =========================================================================
    # GROUPE 1 : GV (Deep Learning MLP) + BOOSTING (LightGBM)
    # =========================================================================
    if RUN_GROUP_1:
        
        logs.write("\n\n")
        logs.write("-"*80+"\n")
        logs.write(f" LOGS DU GROUPE 1 : STRATÉGIES GLOBALES VECTORIELLES (GV) + LightGBM\n")
        logs.write("-"*80)
        logs.write("\n\n\n")

        print(f"\n{'#'*80}")
        print(" DÉMARRAGE GROUPE 1 : STRATÉGIES GLOBALES VECTORIELLES (GV) + LightGBM")
        print(f"{'#'*80}")

        # 1.A. Pipeline GV (MLP)
        gv_experiments = [
            {'entree': 'GV', 'res': '0', 'inter': 'f', 'trials': 150, 'dims': [0, 32, 64, 128, 256]},
            {'entree': 'GV', 'res': '0', 'inter': 'v', 'trials': 150, 'dims': [0, 32, 64, 128, 256]},
            {'entree': 'GV', 'res': '1', 'inter': 'f', 'trials': 150, 'dims': [0, 32, 64, 128, 256]},
            {'entree': 'GV', 'res': '1', 'inter': 'v', 'trials': 150, 'dims': [0, 32, 64, 128, 256]},
        ]
        
        for exp in gv_experiments:
            e, r, i, n_trials = exp['entree'], exp['res'], exp['inter'], exp['trials']
            for dim in exp['dims']:
                suffixe = f"D{dim}" if dim > 0 else "D0"
                model_name = f"{e}_{r}_{i}_{suffixe}"
                model_start = time.time()
                
                logs.write(f"\n\n{'*'*70}\n\n PIPELINE DL (MLP) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}\n\n")
                print(f"\n{'*'*70}\n PIPELINE DL (MLP) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}")

                if dim > 0:
                    logs.write(f"\n   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...\n")
                    print(f"   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...")
                    #optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=10)
                else:
                    logs.write(f"\n   [1/3] Mode D0 : Pas d'Auto-encodeur.\n")
                    print(f"   [1/3] Mode D0 : Pas d'Auto-encodeur.")
                
                logs.write(f"\n   [2/3] Optimisation du Modèle Prédictif...\n")
                print(f"   [2/3] Optimisation du Modèle Prédictif...")
                # optimize(df_train, entree=e, residuelle=r, inter=i, suffixe=suffixe, n_trials=n_trials)
                
                logs.write(f"\n   [3/3] Évaluation Finale...\n")
                print(f"   [3/3] Évaluation Finale...")
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, suffixe=suffixe)
                
                logs.write(f"\n--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---\n")
                print(f"--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---")

        # 1.B. Pipeline LightGBM
        lgbm_experiments = [
            #{'entree': 'GV', 'res': '1', 'inter': 'f', 'dims': [32, 64, 128]},
            #{'entree': 'GV', 'res': '1', 'inter': 'v', 'dims': [32, 64]},
        ]
        
        for exp in lgbm_experiments:
            e, r, i = exp['entree'], exp['res'], exp['inter']
            for dim in exp['dims']:
                suffixe = f"D{dim}"
                logs.write(f"\n\n{'*'*70}\n PIPELINE LIGHTGBM : {e}_{r}_{i}_{suffixe}\n\n{'*'*70}\n")
                print(f"\n{'*'*70}\n PIPELINE LIGHTGBM : {e}_{r}_{i}_{suffixe}\n{'*'*70}")
                
                # Vérification de sécurité (normalement l'AE a déjà été généré par la boucle GV précédente)
                optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=10)
                train_latent_boosting(df_train, df_test, entree=e, residuelle=r, inter=i, latent_dim=dim, suffixe=suffixe)


    # =========================================================================
    # GROUPE 2 : GM (Deep Learning CNN) - INTERMÉDIAIRE 'f'
    # =========================================================================
    if RUN_GROUP_2:

        logs.write("-"*80+"\n"+"-"*80)
        logs.write(f" LOGS DU GROUPE 2 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Forces (f)")
        logs.write("\n\n\n")

        print(f"\n{'#'*80}")
        print(" DÉMARRAGE GROUPE 2 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Forces (f)")
        print(f"{'#'*80}")
        
        gm_f_experiments = [
            {'entree': 'GM', 'res': '1', 'inter': 'f', 'trials': 50, 'dims': [0, 128, 256, 512]}
        ]
        
        for exp in gm_f_experiments:
            e, r, i, n_trials = exp['entree'], exp['res'], exp['inter'], exp['trials']
            for dim in exp['dims']:
                suffixe = f"D{dim}" if dim > 0 else "D0"
                model_name = f"{e}_{r}_{i}_{suffixe}"
                model_start = time.time()
                
                logs.write(f"\n\n{'*'*70}\n\n PIPELINE DL (CNN) : {model_name} | Optuna Trials : {n_trials}\n\n{'*'*70}")
                print(f"\n{'*'*70}\n PIPELINE DL (CNN) : {model_name} | Optuna Trials : {n_trials}\n{'*'*70}")

                if dim > 0:
                    logs.write(f"\n   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...\n")
                    print(f"   [1/3] Vérification/Création Auto-encodeur ({dim} dim)...")
                    # optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=25)
                else:
                    logs.write(f"\n   [1/3] Mode D0 : Pas d'Auto-encodeur.\n")
                    print(f"   [1/3] Mode D0 : Pas d'Auto-encodeur.")
                
                logs.write(f"\n   [2/3] Optimisation du Modèle Prédictif...\n")
                print(f"   [2/3] Optimisation du Modèle Prédictif...")
                # optimize(df_train, entree=e, residuelle=r, inter=i, suffixe=suffixe, n_trials=n_trials)
                
                logs.write(f"\n   [3/3] Évaluation Finale...\n")
                print(f"   [3/3] Évaluation Finale...")
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, suffixe=suffixe)
                
                logs.write(f"\n--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---\n")
                print(f"--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---")


    # =========================================================================
    # GROUPE 3 : GM (Deep Learning CNN) - INTERMÉDIAIRE 'v'
    # =========================================================================
    if RUN_GROUP_3:
                
        logs.write("-"*80+"\n"+"-"*80)
        logs.write(f" LOGS DU GROUPE 3 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Vitesses (v)")
        logs.write("\n\n\n")
        
        print(f"\n{'#'*80}")
        print(" DÉMARRAGE GROUPE 3 : STRATÉGIES GLOBALES MATRICIELLES (GM) - Vitesses (v)")
        print(f"{'#'*80}")
        
        gm_v_experiments = [
            {'entree': 'GM', 'res': '1', 'inter': 'v', 'trials': 50, 'dims': [0, 128, 256, 512]}
        ]
        
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
                    # optimize_and_train_ae(df_train, entree=e, residuelle=r, inter=i, latent_dim=dim, n_trials=25)
                else:
                    logs.write(f"\n   [1/3] Mode D0 : Pas d'Auto-encodeur.\n")
                    print(f"   [1/3] Mode D0 : Pas d'Auto-encodeur.")
                
                logs.write(f"\n   [2/3] Optimisation du Modèle Prédictif...\n")
                print(f"   [2/3] Optimisation du Modèle Prédictif...")
                # optimize(df_train, entree=e, residuelle=r, inter=i, suffixe=suffixe, n_trials=n_trials)
                
                logs.write(f"\n   [3/3] Évaluation Finale...\n")
                print(f"   [3/3] Évaluation Finale...")
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i, suffixe=suffixe)
                
                logs.write(f"\n--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---\n")
                print(f"--- Modèle {model_name} terminé en {format_duration(time.time() - model_start)} ---")

    if RUN_GROUP_1 == True :
        logs.write(f"\n\nSession terminée (groupe 1) en {format_duration(time.time() - global_start)}\n")
    elif RUN_GROUP_2 == True :
        logs.write(f"\n\nSession terminée (groupe 2) en {format_duration(time.time() - global_start)}\n")
    elif RUN_GROUP_3 == True :
        logs.write(f"\n\nSession terminée (groupe 3) en {format_duration(time.time() - global_start)}\n")

    print(f"\nSession terminée en {format_duration(time.time() - global_start)}")

if __name__ == "__main__":
    main()
