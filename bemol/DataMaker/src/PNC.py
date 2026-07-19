import sys
import os
import pathlib as p
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimEnv as sev

castor_loc = p.Path('DataMaker/data_vortex_yaw_mexico')

def Castor_data(yaw, tsr) :
    if float(yaw)/10 > 1 : 
        yawS = '0'/str(yaw)
    else : 
        yawS = '00'/str(yaw)

    if float(tsr)/10 > 1 :
        tsrS = '0'/str(tsr)
    else : 
        tsrS = '00'/str(tsr)

    interest_data = []
    for f in os.listdir(castor_loc) :
        if yawS and tsrS and 'fn' in f.__name__ : 
            interest_data.append(pd.read_csv(castor_loc/f, comment = '#'))
        if yawS and tsrS and 'ft' in f.__name__ :
            interest_data.append(pd.read_csv(castor_loc/f, comment = '#'))
            break

    if len(interest_data) == 0 :
        raise FileExistsError(f"Aucune donnée castor n'a été trouvé pour (yaw,tsr) = ({yaw, tsr}). Arrêt.\n") 
    elif len(interest_data) > 2 :
        raise ValueError("\nTrop de fichiers ont été extrait, arrêt.\n")
    
    print(f"Donnée castor avec \nYaw = {yaw}\nTsr = {tsr}\ntrouvée et retournée.")
    return interest_data

def plot(abs:str, ord:str, yaw, tsr, BEM_data, SVEN_data, CASTOR_data) :

    if BEM_data == None or SVEN_data == None or CASTOR_data == None : 
        raise ValueError(f"\nErreur dans l'import des données. Arrêt.\n")

    if abs not in ['r', 'phi'] : 
        raise ValueError(f"\nAbscisse non valide. Arrêt.\n")
    if ord not in ['Fn', 'Ft', 'V_eff', 'Alpha_deg'] : 
        raise ValueError(f"\nOrdonnée non valide. Arrêt.\n")
    
    if abs == 'r' :
        val_phi = ['3']
        bem_ord    = BEM_data[BEM_data['yaw'] == yaw and BEM_data['tsr'] == tsr]
        sven_ord   = SVEN_data[SVEN_data['yaw'] == yaw and SVEN_data['tsr'] == tsr]
        castor_ord = CASTOR_data

    
