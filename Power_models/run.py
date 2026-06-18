import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

from Power_models.src.evaluator import evaluator_power, evaluate_baseline
from Power_models.src.optimize import optimize_PM, optimize_AE
from Power_models.src.PModels import PowerMLP 
from Power_models.src.dataloaders import get_splits

import torch
import pathlib as P
import pandas as pd
from time import perf_counter

path_data = P.Path('data/raw/fichier_forces.csv')

def run() : 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_full = pd.read_csv(path_data)
    df_train, df_val = get_splits(df_full)

    evaluate_baseline(df_val)
    for entree in ['GVP'] : #['DP', 'GVP', 'GMP']
        for comp in [True, False] :
            for res in ['2', '1', '0', '-1'] :
                model_name = f"{entree}_{res}"
                
                #Création et optimisation du compresseur des efforts BEM
                optimize_AE(df_train, n_trials = 2, force_opt = True)

                ## Optimisation des HP pour les modèles
                print(f"     Début de l'optimisation Optuna pour {model_name}")
                
                start_opt = perf_counter()
                optimize_PM(df_train, entree = entree, res = res, comp = comp,n_trials = 2)
                stop_opt = perf_counter()

                time_exe = round((stop_opt-start_opt)/60, 2)
                print(f"   Fin de l'optimisation Optuna pour {model_name}. Réalisé en {time_exe} min")
                
                print(f"\n      Début de l'entraînement final + cross valditation")
                
                start_eval = perf_counter()
                evaluator_power(df_train, df_val, entree, res, comp)
                stop_eval = perf_counter()

                time_exe = round((stop_eval - start_eval)/60, 2)
                print(f"     Fin de l'évaluation finale pour {model_name} en {time_exe} min")
        
    return


if __name__ == '__main__' :
    run()