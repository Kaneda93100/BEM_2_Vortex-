import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

from Power_models.src.evaluator import evaluator_power, evaluate_baseline
from Power_models.src.optimize import optimize_PM, optimize_AE
from Power_models.src.PModels import PowerMLP, PowerDensityLoss, PowerLoss
from Power_models.src.dataloaders import get_splits

import torch
import pathlib as P
import pandas as pd
from time import perf_counter

path_data = P.Path('data/raw/fichier_forces.csv')

eps_obj     = 1
n_trials    = 1

eps_cv      = 1
eps_train   = 1

def run() : 
    glob_start = perf_counter()

    if os.path.exists('Power_models/performance/recap_score_glob.csv') :
        print("[Info]   Attention : le dossier performance contient déjà un récapitulatif. Lancer ce script signifie écraser ce fichier. Entrez [Y/N] pour continuer.\n")
        decision = input('Y --> lancer run.py, N --> ne pas lancer run.py   ')
        if decision == "N" or decision == "n" :
            print("Arrêt du script.")
            return
        else :
            print("Le script va s'éxécuter.\n\n\n\n")

    df_full = pd.read_csv(path_data)
    df_train, df_val = get_splits(df_full)
    
    evaluate_baseline(df_val)
    for entree in ['DP', 'GVP', 'GMP'] :
        for comp in [False] :
            if (entree == 'DP' or entree == 'GMP') and comp == True :
                print(f"entree == {entree}, ne support pas le mode compression, entrainement ignoré.\n")
                continue
            for res in ['2', '1', '0', '-1'] :
                if (res == '0' or res == '-1') and entree == 'GMP':
                    print(f"entree == {entree}, ne supporte pas le cas où il n'y a pas d'image BEM en entrée. Entraînement ignoré.\n")
                    continue

                model_name = f"{entree}_{res}_{comp}"
                ## Optimisation des HP pour les modèles
                print(f"[Info]   Début de l'optimisation Optuna pour {model_name}")
                
                start_opt = perf_counter()
                optimize_PM(df_train, entree = entree, res = res, comp = comp, n_trials = n_trials, eps_obj = eps_obj)
                stop_opt = perf_counter()

                time_exe = round((stop_opt-start_opt)/60, 2)
                print(f"   Fin de l'optimisation Optuna pour {model_name}. Réalisé en {time_exe} min")
                
                print(f"\n      Début de l'entraînement final + cross valditation")
                
                start_eval = perf_counter()
                evaluator_power(df_train, df_val, entree, res, comp, eps_cv = eps_cv, eps_train = eps_train)
                stop_eval = perf_counter()

                time_exe = round((stop_eval - start_eval)/60, 2)
                print(f"     Fin de l'évaluation finale pour {model_name} en {time_exe} min")
    
    glob_stop = perf_counter()

    print(f"[Info]      Fin de l'entraînement total, réalisé en {round((glob_stop - glob_start)/60, 3)} minute.\n")
    return


if __name__ == '__main__' :
    run()