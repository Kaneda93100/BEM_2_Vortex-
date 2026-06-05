import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler

from src.models import PolarSurrogate

def train_polar_model():
    # 1. Configuration des chemins
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "geometry/airfoils.csv"

    os.makedirs("models/convert_v", exist_ok=True)
    os.makedirs("scalers", exist_ok=True)

    model_save_path = "models/convert_v/polar_surrogate.pth"
    scaler_save_path = "scalers/scaler_surrogate.pkl"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Impossible de trouver le fichier des polaires : {csv_path}")

    # 2. Chargement des données
    df = pd.read_csv(csv_path)
    X_raw = df[['alpha_deg', 'r']].values
    Y_raw = df[['Cl', 'Cd']].values

    # 3. Normalisation (StandardScaler préserve les signes autour de zéro)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    Y_scaled = scaler_Y.fit_transform(Y_raw)

    # Sauvegarde des scalers pour la dénormalisation dans la Loss Physique
    with open(scaler_save_path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_Y": scaler_Y}, f)

    # Conversion en tenseurs PyTorch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32, device=device)

    # 4. Instanciation du modèle importé
    model = PolarSurrogate(device=device).to(device)
    criterion = nn.MSELoss()

    # Utilisation d'AdamW avec un léger Weight Decay pour lisser les gradients
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    # Utilisation delr_scheduler.CosineAnnealingLR 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500)

    # 5. Boucle d'entraînement
    print(f"\n{'='*50}")
    print(" ENTRAÎNEMENT DU MODÈLE DE SUBSTITUTION (POLAIRES)")
    print(f"{'='*50}")
    print(f" -> Données chargées : {len(df)} lignes.")
    print(f" -> Entraînement sur : {device}")

    epochs = 2500
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, Y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 250 == 0:
            print(f"   Époque {epoch+1:4d}/{epochs} | MSE Loss: {loss.item():.7f}")

    # 6. Sauvegarde des poids finaux
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[OK] Sauvegardes effectuées :")
    print(f"   - Poids du modèle -> {model_save_path}")
    print(f"   - Scalers associés -> {scaler_save_path} (Loss finale : {loss.item():.7f})")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    train_polar_model()