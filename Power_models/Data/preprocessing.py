import sys
import os
path = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__),".."), ".."))
sys.path.append(path)
import pathlib as P
import pandas as pd
from Power_models.src.dataloaders import convert_f_to_power

default_path = P.Path('data/raw/fichier_forces.csv')
path_2_power = P.Path('Power_models/Data/DataDir')


def prepro_BasicP(path2data) :
    
    df = pd.read_csv(path2data)
    print(f"    Données chargée depuis l'emplacement {path2data}")

    Power_full = convert_f_to_power(df) 
    Pows_only  = Power_full[['yaw', 'TSR', 'Cp_BEM', 'Cp_SVEN']] 
    Pows_only.to_csv(path_2_power/'Power.csv', index = False)

    print(f"    Données convertie et enregistrée à l'emplacement {path_2_power/'Power.csv'}.\n")
    return

def run_prepro(dir_data = default_path) :
    prepro_BasicP(dir_data)
    return

if __name__ == '__main__' :
    run_prepro()
