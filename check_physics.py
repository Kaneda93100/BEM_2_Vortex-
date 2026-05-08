import numpy as np
import pandas as pd
from src.data_loader import load_clean_data
from src.physics import convert_v_to_f

def main():
    print("Chargement des données brutes (Forces et Vitesses)...")
    # La fonction load_clean_data ne prend plus le fichier induction
    df = load_clean_data(
        path_forces="data/raw/fichier_forces.csv", 
        path_vitesses="data/raw/fichier_vitesses.csv"
    )
    
    print(f"Nombre de points à vérifier : {len(df)}\n")

    # Mise à jour des sources : ajout de BEM_NoYaw
    sources = {
        'BEM': '', 
        'BEM_NoYaw': '_NoYaw', 
        'SVEN': ''
    }

    for source_label, suffix in sources.items():
        print(f"=== VÉRIFICATION DES DONNÉES {source_label.upper()} ===")
        
        # Mapping des colonnes selon les nouvelles entêtes du CSV
        # Format : V_eff_BEM, V_eff_BEM_NoYaw, V_eff_SVEN
        col_v = f'V_eff_{source_label}' if 'NoYaw' not in source_label else f'V_eff_BEM{suffix}'
        col_alpha = f'alpha_{source_label}' if 'NoYaw' not in source_label else f'alpha_BEM{suffix}'
        col_fn = f'Fn_{source_label}' if 'NoYaw' not in source_label else f'Fn_BEM{suffix}'
        col_ft = f'Ft_{source_label}' if 'NoYaw' not in source_label else f'Ft_BEM{suffix}'

        # ---------------------------------------------------------
        # Étape unique : Test de convert_v_to_f (V_eff, alpha -> Fn, Ft)
        # ---------------------------------------------------------
        # On utilise les vitesses et angles d'attaque du CSV pour recalculer les forces
        calc_fn, calc_ft = convert_v_to_f(
            df[col_v].values, 
            df[col_alpha].values, 
            df['r'].values
        )
        
        # Calcul des erreurs par rapport aux forces SVEN/BEM enregistrées
        err_fn = np.mean(np.abs(calc_fn - df[col_fn].values))
        err_ft = np.mean(np.abs(calc_ft - df[col_ft].values))
        
        # Erreur relative par rapport à la force moyenne
        mean_fn_abs = np.mean(np.abs(df[col_fn].values))
        mean_ft_abs = np.mean(np.abs(df[col_ft].values))
        
        rel_fn = (err_fn / mean_fn_abs) * 100 if mean_fn_abs != 0 else 0
        rel_ft = (err_ft / mean_ft_abs) * 100 if mean_ft_abs != 0 else 0

        print(f"[OK] Conversion v -> f (Vitesses vers Forces) :")
        print(f"    Source utilisée : {col_v} / {col_alpha}")
        print(f"    Erreur absolue moyenne Fn : {err_fn:.4f} N/m ({rel_fn:.2f}%)")
        print(f"    Erreur absolue moyenne Ft : {err_ft:.4f} N/m ({rel_ft:.2f}%)\n")

if __name__ == "__main__":
    main()