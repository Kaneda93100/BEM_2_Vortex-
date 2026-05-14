import optuna
import json
import torch
import torch.nn as nn
import os
from sklearn.model_selection import KFold
from .models import TurbineMLP, EnsemblePointNet
from .data_loader import format_data
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def optimize(df_train, entree, residuelle, inter, n_trials=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"{entree}_{residuelle}_{inter}"
    os.makedirs("hyperparametres", exist_ok=True)
    
    # 1. Formatage sur tout le jeu d'entraînement
    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    
    if entree == 'P':
        num_points = 1296
        num_yaws = len(X_full) // num_points
        X_full = X_full.view(num_points, num_yaws, -1)
        Y_full = Y_full.view(num_points, num_yaws, -1)
        
    def objective(trial):
        # 2. Architectures adaptées
        if entree == 'P':
            n_layers = trial.suggest_int('n_layers', 1, 3)     
            n_neurons = trial.suggest_int('n_neurons', 8, 32)  
        else:
            n_layers = trial.suggest_int('n_layers', 2, 8)     
            n_neurons = trial.suggest_int('n_neurons', 64, 512)
            
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        criterion = nn.MSELoss()
        
        split_range = range(num_yaws) if entree == 'P' else range(len(X_full))
        
        # On ajoute enumerate pour récupérer le numéro du fold (1, 2 ou 3)
        for fold, (train_idx, val_idx) in enumerate(kf.split(split_range)):
            if entree == 'P':
                X_tr, Y_tr = X_full[:, train_idx, :], Y_full[:, train_idx, :]
                X_val, Y_val = X_full[:, val_idx, :], Y_full[:, val_idx, :]
                model = EnsemblePointNet(num_points, X_full.shape[2], Y_full.shape[2], n_layers, n_neurons, dropout_rate).to(device)
            else:
                X_tr, Y_tr = X_full[train_idx], Y_full[train_idx]
                X_val, Y_val = X_full[val_idx], Y_full[val_idx]
                model = TurbineMLP(X_full.shape[1], Y_full.shape[1], n_layers, n_neurons, dropout_rate).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            model.train()
            epochs = 150
            
            # --- LA BARRE DE PROGRESSION  ---
            # trial.number donne l'avancement d'Optuna. leave=False nettoie la console après.
            pbar = tqdm(range(epochs), desc=f"Trial {trial.number} | Fold {fold+1}/3", leave=False)
            
            for epoch in pbar:
                optimizer.zero_grad()
                loss = criterion(model(X_tr), Y_tr)
                loss.backward()
                optimizer.step()
                
                # Mise à jour de la loss toutes les 10 époques
                if (epoch + 1) % 10 == 0:
                    pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
                
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), Y_val).item()
            cv_scores.append(val_loss)
            
        return sum(cv_scores) / len(cv_scores)

   
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    
    print(f"   Démarrage de l'optimisation ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)
    
    with open(f"hyperparametres/{model_name}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f" Modèle {model_name} optimisé. Best MSE: {study.best_value:.4f}")