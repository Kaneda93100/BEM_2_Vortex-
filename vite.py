import numpy as np
import pandas as pd
from training.src.data_loader import load_clean_data
from core.physics import compute_cp

def calculate_macro_statistics_pure():
    print("Chargement et nettoyage des données...")
    df = load_clean_data()
    
    print("Calcul de Cp et Ct par point d'opération (yaw, TSR)...")
    # Intégration spatio-temporelle via physics.py
    df_cp_sven = compute_cp(df, col_fn='Fn_SVEN', col_ft='Ft_SVEN')
    df_cp_bem = compute_cp(df, col_fn='Fn_BEM', col_ft='Ft_BEM')
    
    # Fusion des résultats sur la grille opératoire
    df_macro = pd.merge(df_cp_sven, df_cp_bem, on=['yaw', 'TSR'])
    
    cp_sven = df_macro['Cp_SVEN'].values
    ct_sven = df_macro['Ct_SVEN'].values
    cp_bem = df_macro['Cp_BEM'].values
    ct_bem = df_macro['Ct_BEM'].values
    
    # =========================================================================
    # CALCUL DES ERREURS RELATIVES LOCALES PURES
    # =========================================================================
    # Formule brute point par point : |BEM - SVEN| / |SVEN|
    err_rel_cp = np.abs(cp_bem - cp_sven) / np.abs(cp_sven)
    err_rel_ct = np.abs(ct_bem - ct_sven) / np.abs(ct_sven)
    
    # RMSE sur ces erreurs relatives
    rmse_rel_cp = np.sqrt(np.mean(err_rel_cp ** 2))
    rmse_rel_ct = np.sqrt(np.mean(err_rel_ct ** 2))
    
    # =========================================================================
    # AFFICHAGE DES RÉSULTATS
    # =========================================================================
    print("\n" + "="*60)
    print(" STATISTIQUES MACROSCOPIQUES PURES (YAW, TSR)")
    print("="*60)
    print(f" Nombre de configurations uniques (yaw, TSR) : {len(df_macro)}")
    
    print("\n" + "-"*50)
    print(" 📊 COEFFICIENT DE PUISSANCE : C_P")
    print("-" * 50)
    print(f"  [SVEN]  Min: {np.min(cp_sven):.4f} | Max: {np.max(cp_sven):.4f} | Moyenne: {np.mean(cp_sven):.4f}")
    print(f"  [BEM]   RMSE Relative Pure : {rmse_rel_cp * 100:.2f} %")
    print(f"          Erreur Max Locale  : {np.max(err_rel_cp) * 100:.2f} %")
    print(f"          Erreur Moyenne     : {np.mean(err_rel_cp) * 100:.2f} %")
    
    print("\n" + "-"*50)
    print(" 📊 COEFFICIENT DE POUSSÉE : C_T")
    print("-" * 50)
    print(f"  [SVEN]  Min: {np.min(ct_sven):.4f} | Max: {np.max(ct_sven):.4f} | Moyenne: {np.mean(ct_sven):.4f}")
    print(f"  [BEM]   RMSE Relative Pure : {rmse_rel_ct * 100:.2f} %")
    print(f"          Erreur Max Locale  : {np.max(err_rel_ct) * 100:.2f} %")
    print(f"          Erreur Moyenne     : {np.mean(err_rel_ct) * 100:.2f} %")
    print("="*60)

if __name__ == "__main__":
    calculate_macro_statistics_pure()