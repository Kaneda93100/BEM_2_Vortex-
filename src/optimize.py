import optuna
import json
import torch
import torch.nn as nn
import os
from sklearn.model_selection import KFold
from .models import TurbineMLP, TurbineCNN, ConvolutionalAutoencoder, LinearAutoencoder
from .data_loader import format_data
from tqdm import tqdm

def optimize(df_train, entree, residuelle, inter, n_trials=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"{entree}_{residuelle}_{inter}"
    os.makedirs("hyperparametres", exist_ok=True)
    
    # Formatage sur tout le jeu d'entraînement
    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    in_dim = X_full.shape[1] 
    out_dim = Y_full.shape[1] 
    
    def objective(trial):
        # --- CONFIGURATION DES HYPERPARAMÈTRES ---
        if entree == 'GV':
            n_layers = trial.suggest_int('n_layers', 2, 7)     
            n_neurons = trial.suggest_int('n_neurons', 128, 1024, step=64)
            
            use_ae = trial.suggest_categorical('use_autoencoder', [True, False])
            latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256]) if use_ae else 0
            
        elif entree == 'GM':
            n_layers = trial.suggest_int('n_layers', 3, 8)     
            base_filters = trial.suggest_categorical('base_filters', [32, 64, 128])
            
            use_ae = trial.suggest_categorical('use_autoencoder', [True, False])
            latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256]) if use_ae else 0
            
        # Régularisation globale
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        criterion = nn.MSELoss()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(X_full)))):
            X_tr, Y_tr = X_full[train_idx], Y_full[train_idx]
            X_val, Y_val = X_full[val_idx], Y_full[val_idx]
            
            # ==========================================
            # 1. ENTRAÎNEMENT DE L'AUTO-ENCODEUR (Pré-traitement)
            # ==========================================
            if use_ae:
                if entree == 'GM':
                    cae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, device=device).to(device)
                else:
                    cae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
                    
                optimizer_ae = torch.optim.Adam(cae.parameters(), lr=1e-3)
                cae.train()
                
                # Entraînement de l'AE 
                for epoch_ae in range(400): 
                    optimizer_ae.zero_grad()
                    loss_ae = criterion(cae(Y_tr), Y_tr)
                    loss_ae.backward()
                    optimizer_ae.step()
                    
                cae.eval()
                with torch.no_grad():
                    Y_tr_target = cae.encode(Y_tr)
                    Y_val_target = cae.encode(Y_val)
            else:
                # Mode fonction identité
                Y_tr_target, Y_val_target = Y_tr, Y_val
            
            # ==========================================
            # 2. ENTRAÎNEMENT DU MODÈLE PRÉDICTIF PRINCIPAL
            # ==========================================
            if entree == 'GV':
                current_out = latent_dim if use_ae else out_dim
                model = TurbineMLP(in_dim, current_out, n_layers, n_neurons, dropout_rate, device=device).to(device)
            elif entree == 'GM':
                model = TurbineCNN(in_channels=in_dim, out_channels=out_dim, 
                                   use_autoencoder=use_ae, latent_dim=latent_dim,
                                   n_layers=n_layers, base_filters=base_filters, 
                                   dropout_rate=dropout_rate, device=device).to(device)

            # Optimiseur avec L2 Regularization (weight_decay)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Le Scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
            )
            
            epochs = 500 
            best_val_loss = float('inf')
            
            pbar = tqdm(range(epochs), desc=f"Trial {trial.number} | Fold {fold+1}/3", leave=False)
            
            for epoch in pbar:
                # --- Phase d'entraînement ---
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_tr), Y_tr_target) 
                loss.backward()
                optimizer.step()
                
                # --- Phase de validation ---
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val), Y_val_target).item()
                
                # Le scheduler ajuste le LR si la val_loss stagne
                scheduler.step(val_loss)
                
                # On sauvegarde en mémoire la meilleure performance absolue atteinte
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({"Loss_Tr": f"{loss.item():.6f}", "Loss_Val": f"{val_loss:.6f}", "LR": f"{current_lr:.1e}"})
                
            # Optuna reçoit la meilleure perte de validation atteinte durant les 500 époques
            cv_scores.append(best_val_loss)
            
        return sum(cv_scores) / len(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    
    print(f"   Démarrage de l'optimisation ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)
    
    with open(f"hyperparametres/{model_name}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"   Modèle {model_name} optimisé. Best CV MSE: {study.best_value:.6f}")