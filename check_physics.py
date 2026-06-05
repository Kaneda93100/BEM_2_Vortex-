import numpy as np
import pandas as pd
import torch

from src.data_loader import load_clean_data
from src.physics import convert_v_to_f, get_geometry
from src.models import PolarSurrogate, convert_v_to_f_torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Chargement des données brutes sur {device}...")

    df = load_clean_data("data/raw/fichier_forces.csv", "data/raw/fichier_vitesses.csv")
    print(f"Nombre de points à vérifier : {len(df)}\n")

    # Interpolateur PyTorch 1D exact
    polar_surrogate = PolarSurrogate(device=device).to(device)

    geom = get_geometry()
    
    # Reconstruction des vecteurs de rayons et cordes conformes à la forme des données
    r_vals = df['r'].values
    c_vals = np.array([float(geom.get_chord(r)) for r in r_vals], dtype=np.float32)

    r_tensor = torch.tensor(r_vals, dtype=torch.float32, device=device)
    c_tensor = torch.tensor(c_vals, dtype=torch.float32, device=device)

    sources = ['BEM', 'SVEN']
    for source_label in sources:
        print(f"\n=== VÉRIFICATION DES DONNÉES {source_label.upper()} ===")
        
        col_v, col_alpha = f'V_eff_{source_label}', f'alpha_{source_label}'
        col_fn, col_ft = f'Fn_{source_label}', f'Ft_{source_label}'
        
        if col_v not in df.columns:
            print(f"Saut : Colonne {col_v} introuvable.")
            continue
            
        val_fn, val_ft = df[col_fn].values, df[col_ft].values

        # 1. Version NUMPY (version originale non différentiable parPyTorch)
        calc_fn_np, calc_ft_np = convert_v_to_f(df[col_v].values, df[col_alpha].values, df['r'].values)
        
        # 2. Version PYTORCH (Nouvel interpolateur linéaire)
        v_eff_tensor = torch.tensor(df[col_v].values, dtype=torch.float32, device=device)
        alpha_tensor = torch.tensor(df[col_alpha].values, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            f_pred_torch = convert_v_to_f_torch(v_eff_tensor, alpha_tensor, r_tensor, c_tensor, polar_surrogate)
        
        calc_fn_pt = f_pred_torch[..., 0].cpu().numpy()
        calc_ft_pt = f_pred_torch[..., 1].cpu().numpy()

        # Calculs des erreurs relatives et absolues
        for label, pred_fn, pred_ft in [("NUMPY", calc_fn_np, calc_ft_np), ("PYTORCH", calc_fn_pt, calc_ft_pt)]:
            err_fn = np.mean(np.abs(pred_fn - val_fn))
            err_ft = np.mean(np.abs(pred_ft - val_ft))
            
            mean_fn = np.mean(np.abs(val_fn))
            mean_ft = np.mean(np.abs(val_ft))
            
            rel_fn = (err_fn / mean_fn) * 100 if mean_fn > 0 else 0
            rel_ft = (err_ft / mean_ft) * 100 if mean_ft > 0 else 0
            
            print(f" [{label}] :")
            print(f"    Erreur Fn : {err_fn:.4f} N/m ({rel_fn:.2f}%)")
            print(f"    Erreur Ft : {err_ft:.4f} N/m ({rel_ft:.2f}%)")

if __name__ == "__main__":
    main()