import optuna
import json
import torch
import torch.nn as nn
import os
from .models import ConvolutionalAutoencoder, LinearAutoencoder
from .data_loader import format_data
from tqdm import tqdm

def optimize_and_train_ae(df_train, entree, residuelle, inter, latent_dim, n_trials=10):
    """
    Optimise et entraîne spécifiquement un Auto-encodeur.
    Crée 1 des 8 types d'AE possibles selon l'entrée (GV/GM), la résiduelle (0/1) et l'intermédiaire (f/v).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    suffixe = f"D{latent_dim}"
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    
    os.makedirs("hyperparametres", exist_ok=True)
    os.makedirs("models/ae", exist_ok=True)
    
    # Vérification de l'existence des poids directement dans le dossier 'models/ae/'
    ae_weights_path = f"models/ae/ae_{saved_name}.pth"
    if os.path.exists(ae_weights_path):
        print(f"   [INFO] Auto-encodeur {saved_name} déjà existant dans 'models/'. Entraînement ignoré.")
        return

    print(f"\n{'='*50}")
    print(f" CRÉATION AUTO-ENCODEUR : {saved_name}")
    print(f" Type : {'Linéaire (MLP/LightGBM)' if entree == 'GV' else 'Convolutif (CNN)'}")
    print(f"{'='*50}")

    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    out_dim = Y_full.shape[1] 
    criterion = nn.MSELoss()
    
    def objective_ae(trial):
        ae_lr = trial.suggest_float('ae_lr', 1e-4, 5e-3, log=True)
        
        if entree == 'GM':
            ae_depth = trial.suggest_int('ae_depth', 2, 3)
            ae_base_filters = trial.suggest_categorical('ae_base_filters', [16, 32])
            ae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, 
                                          depth=ae_depth, base_filters=ae_base_filters, device=device).to(device)
        else:
            ae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
            
        optimizer_ae = torch.optim.Adam(ae.parameters(), lr=ae_lr, weight_decay=1e-5)
        
        # Entraînement
        ae.train()
        for _ in range(50):
            optimizer_ae.zero_grad()
            loss = criterion(ae(Y_full), Y_full)
            loss.backward()
            optimizer_ae.step()
            
        ae.eval()
        with torch.no_grad():
            val_loss = criterion(ae(Y_full), Y_full).item()
        return val_loss

    # Recherche rapide Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_ae = optuna.create_study(direction='minimize')
    print(f"   [1/2] Recherche Optuna pour l'architecture AE ({n_trials} trials)...")
    study_ae.optimize(objective_ae, n_trials=n_trials)
    
    best_ae_params = study_ae.best_params
    best_ae_params['use_autoencoder'] = True
    best_ae_params['latent_dim'] = latent_dim
    print(f"   -> Meilleurs paramètres trouvés : {best_ae_params}")
    
    # --- Entraînement de l'AE final optimal (500 époques) ---
    print(f"   [2/2] Entraînement de l'AE final (500 époques)...")
    if entree == 'GM':
        final_ae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, 
                                            depth=best_ae_params['ae_depth'], 
                                            base_filters=best_ae_params['ae_base_filters'], 
                                            device=device).to(device)
    else:
        final_ae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
        
    optimizer_final_ae = torch.optim.Adam(final_ae.parameters(), lr=best_ae_params['ae_lr'], weight_decay=1e-5)
    
    final_ae.train()
    pbar = tqdm(range(500), desc="   Training AE", leave=False)
    for epoch in pbar:
        optimizer_final_ae.zero_grad()
        loss = criterion(final_ae(Y_full), Y_full)
        loss.backward()
        optimizer_final_ae.step()
        
        if (epoch + 1) % 50 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

    # --- Fichier JSON Global ---
    json_master_path = "hyperparametres/ae_hyperparameters.json"
    
    # Charger le dictionnaire existant ou en créer un nouveau
    if os.path.exists(json_master_path):
        with open(json_master_path, "r") as f:
            all_ae_params = json.load(f)
    else:
        all_ae_params = {}
        
    # Ajouter/Mettre à jour les paramètres pour ce modèle précis
    all_ae_params[saved_name] = best_ae_params
    
    with open(json_master_path, "w") as f:
        json.dump(all_ae_params, f, indent=4)
        
    # Enregistrement du fichier de poids .pth dans le dossier 'models/'
    torch.save(final_ae.state_dict(), ae_weights_path)
    
    print(f"   [OK] Hyperparamètres ajoutés au dictionnaire {json_master_path}")
    print(f"   [OK] Poids de l'AE sauvegardés dans {ae_weights_path} (Loss finale : {loss.item():.6f})")