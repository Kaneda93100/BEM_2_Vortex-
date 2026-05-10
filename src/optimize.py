import optuna
import json
import torch
import torch.nn as nn
import os
from sklearn.model_selection import KFold
from .models import TurbineMLP
from .data_loader import format_data
from torch.utils.data import DataLoader, TensorDataset

def optimize(df_train, entree, residuelle, inter, n_trials=50):
    # Détection du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = f"{entree}_{residuelle}_{inter}"
    os.makedirs("hyperparametres", exist_ok=True)
    
    # 1. Formatage et normalisation sur tout le jeu d'entraînement.
    X_full, Y_full = format_data(df_train, entree, residuelle, inter, is_train=True)
    
    # Transfert global des données sur le GPU ---
    X_full = X_full.to(device)
    Y_full = Y_full.to(device)
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 2, 8)
        n_neurons = trial.suggest_int('n_neurons', 64, 512)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        criterion = nn.MSELoss()
        
        # 2. Le split se fait sur les index 
        for train_idx, val_idx in kf.split(range(len(X_full))):
            X_tr, Y_tr = X_full[train_idx], Y_full[train_idx]
            X_val, Y_val = X_full[val_idx], Y_full[val_idx]

            if entree == 'G':
                b_size = len(X_tr) # Full batch pour G
            elif entree in ['GR', 'GA']:
                b_size = 32        # Mini-batch adapté pour les profils
            else: # 'L'
                b_size = 128       # Mini-batch classique
                
            train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=b_size, shuffle=True)

            # Modèle transféré sur le GPU ---
            model = TurbineMLP(X_full.shape[1], Y_full.shape[1], n_layers, n_neurons, dropout_rate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            model.train()
            # Simple boucle d'entraînement (réduite pour Optuna)
            for epoch in range(150):
                model.train()
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
                
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), Y_val).item()
            cv_scores.append(val_loss)
            
        return sum(cv_scores) / len(cv_scores) # On MINIMISE la MSE

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Sauvegarde des meilleurs paramètres
    with open(f"hyperparametres/{model_name}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"   Modèle {model_name} optimisé sur {device}. Best MSE: {study.best_value:.4f}")