# training/src/optimize_ae.py
import optuna
import json
import torch
import torch.nn as nn
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

from core.models import ConvolutionalAutoencoder, LinearAutoencoder, TorchScaler
from training.src.data_loader import format_data, get_D_tensor
from training.src.trainer import fit_model
from core.config import EPOCHS_AE, TRIALS_AE, LR_BOUNDS

class AEPhysicalLoss(nn.Module):
    """
    Loss pour l'AE évaluant la reconstruction dans l'espace des forces brutes (N/m),
    puis renormalisant par l'écart-type global pour garder un gradient stable.
    """
    def __init__(self, scaler_Y, scaler_F_abs, D_tensor, entree, is_cnn_for_ae, device):
        super().__init__()
        self.scaler_Y = TorchScaler(scaler_Y, device)
        self.D_tensor = D_tensor.to(device)
        
        # Extraction des stats de normalisation des forces brutes
        mean_abs = torch.tensor(scaler_F_abs.mean_, dtype=torch.float32, device=device)
        scale_abs = torch.tensor(scaler_F_abs.scale_, dtype=torch.float32, device=device)
        

        if is_cnn_for_ae:
            # Format 4D (Batch, 2, 36, 72)
            self.mean_flat = mean_abs.view(1, 2, 1, 1)
            self.scale_flat = scale_abs.view(1, 2, 1, 1)
        else:
            if entree == 'GV':
                # Format GV aplati : données alternées [Fn, Ft, Fn, Ft...]
                repeat_times = self.D_tensor.shape[1] // 2
                self.mean_flat = mean_abs.repeat(repeat_times)
                self.scale_flat = scale_abs.repeat(repeat_times)
            elif entree == 'GM':
                # Format GM aplati : données séquentielles [Fn_1...Fn_N, Ft_1...Ft_N]
                half_len = self.D_tensor.shape[1] // 2
                self.mean_flat = torch.cat([mean_abs[0].repeat(half_len), mean_abs[1].repeat(half_len)])
                self.scale_flat = torch.cat([scale_abs[0].repeat(half_len), scale_abs[1].repeat(half_len)])

    def forward(self, y_pred_norm, y_true_norm):
        # 1. Repasser de l'espace standardisé à l'espace Fn/D, Ft/D
        pred_D = self.scaler_Y.inverse_transform(y_pred_norm)
        true_D = self.scaler_Y.inverse_transform(y_true_norm)
        
        # 2. Forces brutes (N/m) 
        pred_abs = pred_D * self.D_tensor
        true_abs = true_D * self.D_tensor
        
        # 3. Erreur absolue en N/m
        err_abs = pred_abs - true_abs
        
        # 4. RENORMALISATION
        err_norm = (err_abs - self.mean_flat) / self.scale_flat
        
        # 5. MSE sur l'erreur renormalisée
        return nn.functional.mse_loss(err_norm, torch.zeros_like(err_norm))


def optimize_and_train_ae(df_train, entree, residuelle, inter, latent_dim, ae_nature, n_trials=TRIALS_AE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    suffixe = f"D{ae_nature}{latent_dim}"
    base_model_name = f"{entree}_{residuelle}_{inter}"
    saved_name = f"{base_model_name}_{suffixe}"
    
    os.makedirs("training/hyperparametres", exist_ok=True)
    os.makedirs("training/models/ae", exist_ok=True)
    
    ae_weights_path = f"training/models/ae/ae_{saved_name}.pth"
    if os.path.exists(ae_weights_path):
        print(f"   [INFO] Auto-encodeur {saved_name} déjà existant. Entraînement ignoré.")
        return

    _, Y_full_raw = format_data(df_train, entree, residuelle, inter, is_train=True, device=device)
    
    # L'AE CNN manipule des tenseurs 4D, indépendamment de l'entree (GV ou GM)
    is_cnn_for_ae = (ae_nature == 'M')

    if is_cnn_for_ae:
        out_dim = 2
        # FORCE le format 4D (Batch, Canaux=2, R=36, Theta=72) pour le Conv2d
        if Y_full_raw.dim() == 2:
            Y_target = Y_full_raw.view(Y_full_raw.size(0), 2, 36, 72)
        else:
            Y_target = Y_full_raw
    else:
        out_dim = np.prod(Y_full_raw.shape[1:]) 
        Y_target = Y_full_raw.view(Y_full_raw.size(0), -1)
        
    print(f"\n{'='*50}")
    print(f" CRÉATION AUTO-ENCODEUR : {saved_name}")
    print(f" Type : {'Linéaire (MLP)' if ae_nature == 'V' else 'Convolutif (CNN)'}")
    print(f"{'='*50}")

    # PRÉPARATION DE LA LOSS PHYSIQUE RENORMALISÉE (Uniquement si inter == 'f')
    physical_criterion = None
    if inter == 'f':
        # 1. Création du scaler absolu
        fn_abs = df_train['Fn_SVEN'].values.astype(np.float32).reshape(-1, 1)
        ft_abs = df_train['Ft_SVEN'].values.astype(np.float32).reshape(-1, 1)
        scaler_F_abs = StandardScaler()
        scaler_F_abs.fit(np.hstack([fn_abs, ft_abs]))
        
        # 2. Chargement du scaler de l'espace cible
        scaler_path = f"training/scalers/scaler_Y_{entree}_{residuelle}_{inter}.pkl"
        with open(scaler_path, 'rb') as f:
            scaler_Y = pickle.load(f)
            
        # 3. Récupération et FORMATTAGE DU TENSEUR D
        D_tensor = get_D_tensor(df_train, entree, device)
        
        # Adaptation de la géométrie de D_tensor pour correspondre à Y_target
        if is_cnn_for_ae:
            if entree == 'GM':
                # D_tensor est (Batch, 2592) -> on le broadcast sur les 2 canaux (Fn et Ft)
                D_tensor = D_tensor.view(-1, 1, 36, 72).expand(-1, 2, -1, -1)
            elif entree == 'GV':
                # D_tensor est (N, 5184) avec [D,D] entrelacés -> reshape en (N, 2, 36, 72)
                # pour correspondre à Y_target reshapé de la même façon
                D_tensor = D_tensor.reshape(-1, 2, 36, 72)
        else:
            if entree == 'GM':
                # Pour le MLP (DV), GM concatène Fn (2592) puis Ft (2592) -> (Batch, 5184)
                D_tensor = torch.cat([D_tensor, D_tensor], dim=1)
                
        # 4. Instanciation de la loss
        physical_criterion = AEPhysicalLoss(scaler_Y, scaler_F_abs, D_tensor, entree, is_cnn_for_ae, device)

    def objective_ae(trial):
        lr = trial.suggest_float('ae_lr', LR_BOUNDS[0], LR_BOUNDS[1], log=True)
        
        if is_cnn_for_ae:
            depth = trial.suggest_int('ae_depth', 2, 4)
            base_filters = trial.suggest_categorical('ae_base_filters', [8, 16, 32])
            model = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, depth=depth, base_filters=base_filters, device=device).to(device)
            trial.set_user_attr('ae_depth', depth)
            trial.set_user_attr('ae_base_filters', base_filters)
        else:
            model = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
        
        criterion = physical_criterion if physical_criterion is not None else nn.MSELoss()
        
        _, best_val_loss = fit_model(
            model=model, X=Y_target, Y=Y_target, criterion=criterion, 
            epochs=EPOCHS_AE, lr=lr, device=device, show_progress=False, trial=trial
        )
        return best_val_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study_ae = optuna.create_study(direction='minimize', pruner=pruner)
    print(f"   [1/2] Recherche Optuna pour l'architecture AE ({n_trials} trials)...")
    study_ae.optimize(objective_ae, n_trials=n_trials, show_progress_bar=False)

    best_ae_params = study_ae.best_params
    best_ae_params.update({'use_autoencoder': True, 'latent_dim': latent_dim})
    if is_cnn_for_ae:
        best_ae_params['ae_depth'] = study_ae.best_trial.user_attrs['ae_depth']
        best_ae_params['ae_base_filters'] = study_ae.best_trial.user_attrs['ae_base_filters']
        
    print(f"   -> Meilleurs paramètres trouvés : {best_ae_params}")

    print(f"   [2/2] Entraînement de l'AE final ({EPOCHS_AE} époques)...")
    criterion = physical_criterion if physical_criterion is not None else nn.MSELoss()
    
    if is_cnn_for_ae:
        final_ae = ConvolutionalAutoencoder(in_channels=out_dim, latent_dim=latent_dim, depth=best_ae_params['ae_depth'], base_filters=best_ae_params['ae_base_filters'], device=device).to(device)
    else:
        final_ae = LinearAutoencoder(in_features=out_dim, latent_dim=latent_dim, device=device).to(device)
        
    final_ae, final_loss = fit_model(
        model=final_ae, X=Y_target, Y=Y_target, criterion=criterion, 
        epochs=EPOCHS_AE, lr=best_ae_params['ae_lr'], device=device, show_progress=True
    )

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
    print(f"   [OK] Poids sauvegardés. Loss finale normalisée : {final_loss:.6f}")