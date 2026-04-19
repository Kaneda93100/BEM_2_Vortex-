import optuna
import json
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from .models import TurbineMLP
from .data_loader import format_data

def optimize(df_train, entree, residuelle, inter, n_trials=50):
    
    X, Y = format_data(df_train, entree, residuelle, inter)
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 5)
        n_neurons = trial.suggest_int('n_neurons', 16, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        criterion = nn.MSELoss()
        
        for train_idx, val_idx in kf.split(X):
            X_tr, Y_tr = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
            
            model = TurbineMLP(X.shape[1], Y.shape[1], n_layers, n_neurons, dropout_rate)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Simple boucle d'entraînement (réduite pour Optuna)
            for epoch in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_tr), Y_tr)
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                val_loss = criterion(model(X_val), Y_val).item()
            cv_scores.append(val_loss)
            
        return sum(cv_scores) / len(cv_scores) # On MINIMISE la MSE

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Sauvegarde des meilleurs paramètres
    model_name = f"{entree}_{residuelle}_{inter}"
    with open(f"hyperparametres/{model_name}.json", "w") as f:
        json.dump(study.best_params, f)
        
    print(f"Modèle {model_name} optimisé. Best MSE: {study.best_value}")