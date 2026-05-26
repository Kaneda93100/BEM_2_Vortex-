import optuna
import json
import torch
import torch.nn as nn
import os
from sklearn.model_selection import KFold
from .models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder
from .data_loader import format_data
from tqdm import tqdm

def optimize(df_train, entree, residuelle, inter, suffixe, n_trials=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    
    os.makedirs("hyperparametres", exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f" OPTIMISATION MODÈLE PRÉDICTIF : {saved_name}")
    print(f"{'='*50}")
    
    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    criterion = nn.MSELoss()
    
    # --- 1. Gestion du Suffixe et Chargement de l'Auto-encodeur ---
    if suffixe == 'D0':
        use_ae = False
        latent_dim = 0
        ae_params = {"use_autoencoder": False, "latent_dim": 0}
        Y_train_target = Y_full
    else:
        ae_params_path = f"hyperparametres/ae_{saved_name}_params.json"
        ae_weights_path = f"models/ae_{saved_name}.pth" 
        
        use_ae = os.path.exists(ae_params_path) and os.path.exists(ae_weights_path)
        
        if use_ae:
            with open(ae_params_path, "r") as f:
                ae_params = json.load(f)
            latent_dim = ae_params['latent_dim']
            
            # Reconstruction de l'AE pour projeter Y_full dans l'espace latent
            if entree == 'GM':
                ae_model = ConvolutionalAutoencoder(in_channels=Y_full.shape[1], latent_dim=latent_dim,
                                                    depth=ae_params['ae_depth'], base_filters=ae_params['ae_base_filters'], device=device).to(device)
            else:
                ae_model = LinearAutoencoder(in_features=Y_full.shape[1], latent_dim=latent_dim, device=device).to(device)
                
            ae_model.load_state_dict(torch.load(ae_weights_path, map_location=device))
            ae_model.eval()
            with torch.no_grad():
                Y_train_target = ae_model.encode(Y_full)
            print(f"   -> AE chargé avec succès. Espace latent : {latent_dim} dimensions.")
        else:
            print(f"   [ATTENTION] AE requis ({saved_name}) mais introuvable. Mode sans AE forcé.")
            use_ae = False
            latent_dim = 0
            ae_params = {"use_autoencoder": False, "latent_dim": 0}
            Y_train_target = Y_full

    # --- 2. Objectif Optuna pour le Modèle Prédictif ---
    def objective_model(trial):
        # Hyperparamètres communs
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.4)
        n_layers = trial.suggest_int('n_layers', 2, 5)
        
        if entree == 'GV':
            n_neurons = trial.suggest_int('n_neurons', 128, 512, step=64)
        elif entree == 'GM':
            base_filters = trial.suggest_categorical('base_filters', [16, 32, 64])

        # Cross-validation (3 Folds)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_full.cpu().numpy())):
            X_tr = X_full[train_idx]
            Y_tr = Y_train_target[train_idx]
            X_val = X_full[val_idx]
            Y_val = Y_train_target[val_idx]
            
            if entree == 'GV':
                current_out = latent_dim if use_ae else Y_full.shape[1]
                model = TurbineMLP(X_full.shape[1], current_out, n_layers, n_neurons, dropout_rate, device=device).to(device)
            elif entree == 'GM':
                target_dim = latent_dim if use_ae else Y_full.shape[1]
                model = TurbineCNN(in_channels=X_full.shape[1], out_channels=target_dim, 
                                   use_autoencoder=use_ae, latent_dim=latent_dim,
                                   n_layers=n_layers, base_filters=base_filters, dropout_rate=dropout_rate, device=device).to(device)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
            
            best_val_loss = float('inf')

            for epoch in range(150):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_tr), Y_tr) 
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val), Y_val).item()
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
            cv_scores.append(best_val_loss)
            
        return sum(cv_scores) / len(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_model = optuna.create_study(direction='minimize')
    study_model.optimize(objective_model, n_trials=n_trials, show_progress_bar=True)
    
    # --- 3. Fusion et Sauvegarde des Hyperparamètres ---
    final_params = {**ae_params, **study_model.best_params}
    
    with open(f"hyperparametres/{saved_name}.json", "w") as f:
        json.dump(final_params, f, indent=4)
        
    print(f"   [OK] Modèle {saved_name} optimisé. Meilleure CV MSE : {study_model.best_value:.6f}")