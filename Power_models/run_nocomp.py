import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

from Power_models.src.evaluator import evaluator_power, evaluate_baseline
from Power_models.src.optimize import optimize_PM, optimize_AE
from Power_models.src.PModels import PowerMLP, PowerDensityLoss
from Power_models.src.dataloaders import get_splits

import torch
import pathlib as P
import pandas as pd
from time import perf_counter

path_data = P.Path('data/raw/fichier_forces.csv')

def run_nocomp() : 

    df_full = pd.read_csv(path_data)
    df_train, df_val = get_splits(df_full)

    evaluate_baseline(df_val)
    for res in ['2', '1'] :
        model_name = f'GVP_{res}_False'
        
    return


if __name__ == '__main__':
    run_nocomp()