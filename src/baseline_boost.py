import os
import json
import pickle
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from scipy.stats import wasserstein_distance
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# --- FILTRE DES WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

from .models import LinearAutoencoder
from .data_loader import format_data
from .physics import convert_v_to_f
from .evaluate import reconstruct_predictions

def optimize_lgbm(X_train, Z_train, saved_name, n_trials=50):
    """ Optimise et SAUVEGARDE les hyperparamètres de LightGBM dans le dictionnaire centralisé """
    os.makedirs("hyperparametres", exist_ok=True)
    hp_master_path = "hyperparametres/lgbm_hyperparameters.json"
    
    # Si le dictionnaire global existe et contient déjà notre modèle, on le charge
    if os.path.exists(hp_master_path):
        with open(hp_master_path, "r") as f:
            all_lgbm_hps = json.load(f)
        if saved_name in all_lgbm_hps:
            print(f"      -> Chargement des hyperparamètres depuis : {hp_master_path}")
            return all_lgbm_hps[saved_name]
    else:
        all_lgbm_hps = {}
            
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
        
        from sklearn.model_selection import cross_val_score
        model = MultiOutputRegressor(LGBMRegressor(**params, n_jobs=1, verbose=-1))
        # Validation croisée sur les données latentes
        scores = cross_val_score(model, X_train, Z_train, cv=3, scoring='neg_mean_squared_error')
        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    
    print(f"\n   [Lancement de l'optimisation Optuna ({n_trials} trials) pour {saved_name}...]")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    
    # Mise à jour et sauvegarde dans le dictionnaire centralisé
    all_lgbm_hps[saved_name] = best_params
    with open(hp_master_path, "w") as f:
        json.dump(all_lgbm_hps, f, indent=4)
        
    print(f"      -> Paramètres ajoutés au dictionnaire {hp_master_path}")
    return best_params


def train_latent_boosting(df_train, df_test, entree='GV', residuelle=1, inter='f', latent_dim=64, suffixe='D64'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    boost_name = f"{saved_name}_LightGBM_Latent"
    
    recap_path = "performance/recap_scores_globaux.csv"
    

    ae_weights_path = f"models/ae/ae_{saved_name}.pth"
    os.makedirs("models/LightGBM", exist_ok=True)
    boost_model_path = f"models/LightGBM/model_{boost_name}.pkl"
    
    print(f"\n{'='*50}")
    print(f" RUN LATENT BOOSTING : {boost_name}")
    print(f"{'='*50}")
    
    # 1. Préparation données
    X_train, Y_train = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    X_test, Y_test = format_data(df_test, entree, residuelle, inter, is_train=False, device=device)
    
    # 2. Chargement AE 
    if not os.path.exists(ae_weights_path):
        print(f"      [ERREUR] Impossible de trouver l'AE pré-calculé à l'adresse : {ae_weights_path}")
        print("      Assure-toi d'exécuter la phase optimize_and_train_ae avant.")
        return
        
    ae = LinearAutoencoder(in_features=Y_train.shape[1], latent_dim=latent_dim, device=device).to(device)
    ae.load_state_dict(torch.load(ae_weights_path, map_location=device))
    ae.eval()
    
    with torch.no_grad():
        Z_train = ae.encode(Y_train).cpu().numpy()
        
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()

    # 3. OPTIMISATION + ENTRAÎNEMENT 
    print("\n[2/3] Gestion des arbres de décision (LightGBM)...")
    if os.path.exists(boost_model_path):
        print(f"      -> Chargement du modèle LightGBM existant : {boost_model_path}")
        with open(boost_model_path, "rb") as f:
            booster = pickle.load(f)
    else:
        best_params = optimize_lgbm(X_train_np, Z_train, saved_name, n_trials=50) 
        
        booster = MultiOutputRegressor(LGBMRegressor(**best_params, verbose=-1))
        booster.fit(X_train_np, Z_train)
        print("      -> Arbres finaux entraînés.")

    # 4. Inférence sur le set de test et Décodage inverse
    print("\n[3/3] Prédiction, Décodage et Évaluation physique...")
    Z_pred = booster.predict(X_test_np)
    
    Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(device)
    with torch.no_grad():
        Y_pred_norm = ae.decode(Z_pred_tensor).cpu().numpy()

    # 5. Dénormalisation via le Scaler Y
    with open(f"scalers/scaler_Y_{base_model_name}.pkl", 'rb') as f: 
        scaler_Y = pickle.load(f)
    preds_denorm = scaler_Y.inverse_transform(Y_pred_norm)
    
    # 6. Alignement physique topologique
    df_res = reconstruct_predictions(df_test, preds_denorm, entree, residuelle, inter)
    if inter == 'v':
        df_res['Fn_pred'], df_res['Ft_pred'] = convert_v_to_f(
            df_res['V_eff_pred'].values, df_res['alpha_pred'].values, df_res['r'].values
        )
    
    # 7. Calcul des métriques globales
    Fn_s, Ft_s = df_res['Fn_SVEN'].values, df_res['Ft_SVEN'].values
    Fn_p, Ft_p = df_res['Fn_pred'].values, df_res['Ft_pred'].values
    
    rmse_fn = np.sqrt(np.mean((Fn_p - Fn_s)**2))
    rmse_ft = np.sqrt(np.mean((Ft_p - Ft_s)**2))
    rel_fn = (rmse_fn / np.mean(np.abs(Fn_s))) * 100 if np.mean(np.abs(Fn_s)) != 0 else 0
    rel_ft = (rmse_ft / np.mean(np.abs(Ft_s))) * 100 if np.mean(np.abs(Ft_s)) != 0 else 0
    
    score_total = rel_fn + rel_ft
    wd_score = wasserstein_distance(Fn_s, Fn_p) + wasserstein_distance(Ft_s, Ft_p)
    
    results_detail = {
        "Modele": boost_name,
        "Strat_Entree": entree,
        "Residuelle": residuelle,
        "Intermediaire": inter,
        "Suffixe": suffixe,
        "RMSE_Fn": round(rmse_fn, 4),
        "RMSE_Ft": round(rmse_ft, 4),
        "Rel_Fn (%)": round(rel_fn, 2),
        "Rel_Ft (%)": round(rel_ft, 2),
        "Total_Score (%)": round(score_total, 2),
        "Wasserstein_Dist": round(wd_score, 4)
    }
    
    # 8. Écriture dans le récapitulatif global
    os.makedirs("performance", exist_ok=True)
    if os.path.exists(recap_path):
        df_recap = pd.read_csv(recap_path)
        df_recap = df_recap[df_recap["Modele"] != boost_name]
        df_recap = pd.concat([df_recap, pd.DataFrame([results_detail])], ignore_index=True)
    else:
        df_recap = pd.DataFrame([results_detail])
        
    df_recap.to_csv(recap_path, index=False)
    print(f"   Score ajouté au récapitulatif global. Erreur Totale : {score_total:.2f}%")
    
    # Sauvegarde conditionnelle du modèle d'arbres à 16% dans le sous-dossier dédié
    if score_total < 16.0:
        with open(boost_model_path, "wb") as f:
            pickle.dump(booster, f)
        print(f"   [SAUVEGARDE] Performance excellente ({score_total:.2f}% < 16%). Modèle enregistré dans {boost_model_path}")
    else:
        print(f"   [INFO] Score de {score_total:.2f}% >= 16%. Le booster final n'a pas été conservé.")
    print(f"{'='*50}\n")