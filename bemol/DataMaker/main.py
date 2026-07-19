import sys
import os
import time 
import pathlib as P

sys.path.append('/home/arthur/Documents/GitHub/BEM_2_Vortex-/src')

from src import SimEnv as senv
import bemol as bem
from src.lhs_ext import rescale, lhs_extenderV1 as lhs_ext

from scipy.stats.qmc import LatinHypercube as lhc
import numpy as np
import pandas as pd


data_dir = P.Path('bemol/DataMaker/BEMoutputs')
data_merged = P.Path('DataMaker/merged'); os.makedirs(data_merged, exist_ok = True)
sven_dir = P.Path("DataMaker/new_data_sven")
#[0.0018727 , 0.00975357]

def create_BEM_data(list_models = ['IFPEN']) :

    ext_samp = lhs_ext(old_seed = 42, n_samples = 100)
    scaled_samples = rescale(ext_samp)

    tsrs = list(scaled_samples[100:,0]); yaws = list(scaled_samples[100:,1])

    corrections = [
    bem.secondary.hubTipLoss.Prandtl,
    bem.secondary.skewAngle.Burton,
    bem.secondary.turbulentWakeState.Buhl,
    ]
    
    sims = []
    for YM in [ 'IFPEN'] : 
        load_models = corrections.copy()
        load_models.append(getattr(bem.secondary.yawModel,YM))
        sims.append(senv.SimEnv(omega = 44.5163679, U = 12.520228472, yaw = 0, skew = 0, corrections = load_models)) ## !! S'assurer que régler yaw = skew = 0 n'impacte pas les calculs
        load_models = []
        save_filepath = P.Path(data_dir / f'BEM_{YM}.csv')
        sims[-1].para_data_maker(yaws = yaws, tsrs = tsrs, nbr_az = 72, export = save_filepath)

    return 1

def merge_SVEN_BEM() :
    
    sven_df = format_sven(sven_dir)
    
    sven_f = sven_df[['Fn','Ft']]
    sven_v = sven_df[['V_eff', 'Alpha_deg']]

    for data in os.listdir(data_dir) :
        name_file = P.Path(data_dir/data).stem
        df = pd.read_csv(data_dir/data)
        
        df = df.sort_values(by = 'yaw')
        df1 = df.copy()

        for i, yaw in enumerate(sorted(df['yaw'].unique())) :
            df_loc = df1[df['yaw'] == yaw]
            df_loc = df_loc.sort_values(by = 'r')
            df_loc_copy = df_loc.copy()
            for j,r in enumerate(sorted(df['r'].unique())) :
                df_loc_loc = df_loc_copy[df_loc_copy['r'] == r]
                df_loc_loc = df_loc_loc.sort_values(by = 'theta')
                df_loc[72*j:72*(j+1)] = df_loc_loc
            df[2592*i:2592*(i+1)] = df_loc

        df = df.rename(columns = {'Fn' : 'Fn_BEM', 'Ft' : 'Ft_BEM', 'V_eff' : 'V_eff_BEM', 'Alpha_deg' : 'alpha_BEM'})
        df = df.reset_index(drop = True)

        v = ['yaw', 'TSR', 'r', 'theta', 'V_eff_BEM', 'alpha_BEM']
        f = ['yaw', 'TSR', 'r', 'theta', 'Fn_BEM', 'Ft_BEM']

        df_v = df[v]
        df_v[['V_eff_SVEN', 'alpha_SVEN']] = sven_v
        
        df_f = df[f]
        df_f[['Fn_SVEN', 'Ft_SVEN']] = sven_f
        
        df_v.to_csv(f'DataMaker/merged/vitesses.csv', index  = False)
        df_f.to_csv(f'DataMaker/merged/forces.csv', index = False)
    
    return 1


def format_sven(path_sven) :
    
    ## 1. Fusion des .csv contenant les données calculée par SVEN

    cols = ['Yaw', 'TSR', 'r', 'theta', 'Fn', 'Ft', 'V_eff', 'Alpha_deg']
    df = pd.DataFrame(columns = cols)
    for csv in os.listdir(path_sven) :
        df_temp = pd.read_csv(path_sven/P.Path(csv))
        df = pd.merge(df_temp, df, how = 'outer', on = cols)
        df_temp = None

    return df

if __name__ == '__main__' :
    #create_BEM_data()
    #format_sven(P.Path("DataMaker/new_data_sven"))
    merge_SVEN_BEM()
    