import optuna
import json
import torch
import torch.nn as nn
import os
from core.models import ConvolutionalAutoencoder, LinearAutoencoder
from training.src.data_loader import format_data
from training.src.trainer import fit_model
from core.config import EPOCHS_AE, TRIALS_AE, LR_BOUNDS

def optimize_and_train_ae(df_train, entree, residuelle, inter, latent_dim, n_trials=TRIALS_AE):
    """
    Optimise et entraîne spécifiquement un Auto-encodeur.
    Délègue l'entraînement au trainer.py centralisé.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    suffixe = f"D{latent_dim}"
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    
    os.makedirs("training/hyperparametres", exist_ok=True)
    os.makedirs("training/models/ae", exist_ok=True)
    
    ae_weights_path = f"training/models/ae/ae_{saved_name}.pth"
    if os.path.exists(ae_weights_path):
        print(f"   [INFO] Auto-encodeur {saved_name} déjà existant dans 'training/models/'. Entraînement ignoré.")
        return

    print(f"\n{'='*50}")
    print(f" CRÉATION AUTO-ENCODEUR : {saved_name}")
    print(f" Type : {'Linéaire (MLP/LightGBM)' if entree == 'GV' else 'Convolutif (CNN)'}")
    print(f"{'='*50}")

    _, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    out_dim = Y_full.shape[1] 
    criterion = nn.MSELoss()
    
    def objective_ae(trial):
        # Utilisation des hyperparamètres centralisés dans config.py
        ae_lr = trial.suggest_float('ae_lr', LR_BOUNDS[0], LR_BOUNDS[1], log=True)
        
        if entree == 'GM':
            ae_depth = trial.suggest_int('ae_depth', 2, 3)
            ae_base_filters = trial.suggest_categorical('ae_base_filters', [16, 32])
            ae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, 
                                          depth=ae_depth, base_filters=ae_base_filters, device=device).to(device)
        else:
            ae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
            
        # Entraînement rapide Optuna (50 époques)
        _, best_loss = fit_model(
            model=ae, X=Y_full, Y=Y_full, criterion=criterion, 
            epochs=50, lr=ae_lr, device=device, show_progress=False
        )
        
        return best_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_ae = optuna.create_study(direction='minimize')
    print(f"   [1/2] Recherche Optuna pour l'architecture AE ({n_trials} trials)...")
    study_ae.optimize(objective_ae, n_trials=n_trials)
    
    best_ae_params = study_ae.best_params
    best_ae_params['use_autoencoder'] = True
    best_ae_params['latent_dim'] = latent_dim
    print(f"   -> Meilleurs paramètres trouvés : {best_ae_params}")
    
    # --- Entraînement de l'AE final optimal ---
    print(f"   [2/2] Entraînement de l'AE final ({EPOCHS_AE} époques)...")
    if entree == 'GM':
        final_ae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, 
                                            depth=best_ae_params['ae_depth'], 
                                            base_filters=best_ae_params['ae_base_filters'], 
                                            device=device).to(device)
    else:
        final_ae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
        
    # Entraînement complet
    final_ae, final_loss = fit_model(
        model=final_ae, X=Y_full, Y=Y_full, criterion=criterion, 
        epochs=EPOCHS_AE, lr=best_ae_params['ae_lr'], device=device, show_progress=True
    )

    # --- Fichier JSON Global ---
    json_master_path = "training/hyperparametres/ae_hyperparameters.json"
    
    if os.path.exists(json_master_path):
        with open(json_master_path, "r") as f:
            all_ae_params = json.load(f)
    else:
        all_ae_params = {}
        
    all_ae_params[saved_name] = best_ae_params
    
    with open(json_master_path, "w") as f:
        json.dump(all_ae_params, f, indent=4)
        
    torch.save(final_ae.state_dict(), ae_weights_path)
    
    print(f"   [OK] Hyperparamètres ajoutés au dictionnaire {json_master_path}")
    print(f"   [OK] Poids de l'AE sauvegardés dans {ae_weights_path} (Loss finale : {final_loss:.6f})")