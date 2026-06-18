import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from core.config import RHO, KERNEL_SIZE, PADDING_R, PADDING_THETA  # <-- Centralisation config !

class TurbineMLP(nn.Module):
    """ Stratégie GV : Le réseau global vectoriel (MLP classique) """
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, dropout_rate, device='cpu'):
        super(TurbineMLP, self).__init__()
        layers = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_neurons, device=device))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_neurons
        layers.append(nn.Linear(in_features, output_dim, device=device))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PeriodicPadding2d(nn.Module):
    """ Padding physique respectant la périodicité de la grille polaire """
    def __init__(self, pad_r, pad_theta):
        super().__init__()
        self.pad_r = pad_r
        self.pad_theta = pad_theta

    def forward(self, x):
        x = F.pad(x, (self.pad_theta, self.pad_theta, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad_r, self.pad_r), mode='replicate')
        return x


class ResBlockPeriodic(nn.Module):
    """ Bloc Résiduel respectant la périodicité de la grille polaire """
    def __init__(self, in_channels, out_channels, device='cpu'):
        super(ResBlockPeriodic, self).__init__()
        self.pad = PeriodicPadding2d(pad_r=PADDING_R, pad_theta=PADDING_THETA)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=0, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE, padding=0, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, device=device),
                nn.BatchNorm2d(out_channels, device=device)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.pad(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class TurbineCNN(nn.Module):
    """ Stratégie GM : Réseau Convolutif pour prédictions spatiales (36x72) """
    def __init__(self, in_channels, out_channels, use_autoencoder=False, latent_dim=64, n_layers=4, base_filters=64, dropout_rate=0.1, device='cpu'):
        super(TurbineCNN, self).__init__()
        self.use_autoencoder = use_autoencoder

        self.initial_conv = nn.Sequential(
            PeriodicPadding2d(pad_r=PADDING_R, pad_theta=PADDING_THETA),
            nn.Conv2d(in_channels, base_filters, kernel_size=KERNEL_SIZE, padding=0, device=device),
            nn.BatchNorm2d(base_filters, device=device),
            nn.ReLU()
        )
        
        layers = []
        curr_filters = base_filters
        for _ in range(n_layers):
            layers.append(ResBlockPeriodic(curr_filters, curr_filters, device=device))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
                
        self.conv_backbone = nn.Sequential(*layers)
        
        if use_autoencoder:
            self.final_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(curr_filters * 36 * 72, latent_dim, device=device)
            )
        else:
            self.final_head = nn.Sequential(
                PeriodicPadding2d(pad_r=PADDING_R, pad_theta=PADDING_THETA),
                nn.Conv2d(curr_filters, out_channels, kernel_size=KERNEL_SIZE, padding=0, device=device)
            )

    def forward(self, x):
        x = self.initial_conv(x)  
        x = self.conv_backbone(x)
        return self.final_head(x)


class ConvolutionalAutoencoder(nn.Module):
    """ Auto-encodeur Convolutif pour la compression de grilles """
    def __init__(self, in_channels=2, latent_dim=64, depth=2, base_filters=16, device='cpu'):
        super(ConvolutionalAutoencoder, self).__init__()
        self.device = device
        
        enc_layers = []
        curr_in = in_channels
        curr_f = base_filters
        self.target_shapes = []
        h_current, w_current = 36, 72 
        
        for i in range(depth):
            self.target_shapes.append((h_current, w_current))
            enc_layers.append(PeriodicPadding2d(pad_r=PADDING_R, pad_theta=PADDING_THETA))
            enc_layers.append(nn.Conv2d(curr_in, curr_f, kernel_size=KERNEL_SIZE, stride=2, padding=0, device=device))
            enc_layers.append(nn.BatchNorm2d(curr_f, device=device))
            enc_layers.append(nn.ReLU())
            enc_layers.append(ResBlockPeriodic(curr_f, curr_f, device=device))
            
            curr_in = curr_f
            curr_f *= 2 
            h_current = (h_current - 1) // 2 + 1
            w_current = (w_current - 1) // 2 + 1
            
        self.encoder_conv = nn.Sequential(*enc_layers)
        self.curr_channels = curr_in
        self.h_final = h_current
        self.w_final = w_current
        self.flatten_dim = self.curr_channels * self.h_final * self.w_final
        self.encoder_linear = nn.Linear(self.flatten_dim, latent_dim, device=device)

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim, device=device),
            nn.ReLU()
        )
        
        dec_layers = []
        curr_in_dec = self.curr_channels
        restore_shapes = list(reversed(self.target_shapes))
        
        for i in range(depth):
            dec_layers.append(ResBlockPeriodic(curr_in_dec, curr_in_dec, device=device))
            next_f = curr_in_dec // 2 if i < depth - 1 else in_channels
            target_h, target_w = restore_shapes[i]
            
            dec_layers.append(nn.Upsample(size=(target_h, target_w), mode='nearest'))
            dec_layers.append(PeriodicPadding2d(pad_r=PADDING_R, pad_theta=PADDING_THETA))
            dec_layers.append(nn.Conv2d(curr_in_dec, next_f, kernel_size=KERNEL_SIZE, padding=0, device=device))
            
            if i < depth - 1:
                dec_layers.append(nn.BatchNorm2d(next_f, device=device))
                dec_layers.append(nn.ReLU())
            curr_in_dec = next_f

        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x):
        features = self.encoder_conv(x)
        features = features.view(features.size(0), -1)
        return self.encoder_linear(features)

    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, self.curr_channels, self.h_final, self.w_final)
        return self.decoder_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class LinearAutoencoder(nn.Module):
    """ Auto-encodeur (MLP) pour la stratégie GV """
    def __init__(self, in_features=5184, latent_dim=64, device='cpu'):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(512, device=device),
            nn.Linear(512, latent_dim, device=device)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(512, device=device),
            nn.Linear(512, in_features, device=device)
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decoder(self.encoder(x))


class TorchScaler:
    """ Transforme un StandardScaler sklearn en opérations PyTorch différentiables """
    def __init__(self, sklearn_scaler, device):
        self.mean = torch.tensor(sklearn_scaler.mean_, dtype=torch.float32, device=device)
        self.scale = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32, device=device)
        
    def inverse_transform(self, tensor_norm):
        orig_shape = tensor_norm.shape
        tensor_flat = tensor_norm.reshape(orig_shape[0], -1)
        tensor_phys = tensor_flat * self.scale + self.mean
        return tensor_phys.reshape(orig_shape)
        
    def transform(self, tensor_phys):
        orig_shape = tensor_phys.shape
        tensor_flat = tensor_phys.reshape(orig_shape[0], -1)
        tensor_norm = (tensor_flat - self.mean) / self.scale
        return tensor_norm.reshape(orig_shape)


class DecoderLoss(nn.Module):
    """ Loss pour la stratégie 'f' : MSE pure dans l'espace standardisé des coefficients """
    def __init__(self, ae_model=None):
        super().__init__()
        self.ae_model = ae_model
        self.mse = nn.MSELoss()

    def forward(self, z_pred, y_true_norm):
        y_pred_norm = self.ae_model.decode(z_pred) if self.ae_model is not None else z_pred
        return self.mse(y_pred_norm, y_true_norm)


class PolarSurrogate(nn.Module):
    """ Interpolateur 1D vectorisé par section de pale, 100% PyTorch. """
    def __init__(self, device='cpu', csv_path="data/geometry/airfoils.csv"):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.alphas_np = np.sort(df['alpha_deg'].unique())
        self.radii_np = np.sort(df['r'].unique())
        
        df_cl = df.pivot(index='alpha_deg', columns='r', values='Cl')
        df_cd = df.pivot(index='alpha_deg', columns='r', values='Cd')
        
        self.register_buffer('alphas_grid', torch.tensor(self.alphas_np, dtype=torch.float32, device=device))
        self.register_buffer('cl_grid', torch.tensor(df_cl.values, dtype=torch.float32, device=device))
        self.register_buffer('cd_grid', torch.tensor(df_cd.values, dtype=torch.float32, device=device))
        self.register_buffer('radii_grid', torch.tensor(self.radii_np, dtype=torch.float32, device=device))

    def forward(self, alpha_deg, r_tensor):
        shape_orig = alpha_deg.shape
        alpha_flat = alpha_deg.contiguous().flatten()
        r_flat = r_tensor.contiguous().flatten()
        
        dists = torch.abs(r_flat.unsqueeze(1) - self.radii_grid.unsqueeze(0))
        idx_r_flat = torch.argmin(dists, dim=1) 
        
        idx_alpha_high = torch.bucketize(alpha_flat, self.alphas_grid)
        idx_alpha_high = torch.clamp(idx_alpha_high, min=1, max=len(self.alphas_grid) - 1)
        idx_alpha_low = idx_alpha_high - 1
        
        x0 = self.alphas_grid[idx_alpha_low]
        x1 = self.alphas_grid[idx_alpha_high]
        t = (alpha_flat - x0) / (x1 - x0)
        
        y0_cl = self.cl_grid[idx_alpha_low, idx_r_flat]
        y1_cl = self.cl_grid[idx_alpha_high, idx_r_flat]
        y0_cd = self.cd_grid[idx_alpha_low, idx_r_flat]
        y1_cd = self.cd_grid[idx_alpha_high, idx_r_flat]
        
        cl_interp = y0_cl + t * (y1_cl - y0_cl)
        cd_interp = y0_cd + t * (y1_cd - y0_cd)
        return torch.stack([cl_interp.view(shape_orig), cd_interp.view(shape_orig)], dim=-1)


def convert_v_to_f_torch(v_eff, alpha_deg, r_tensor, c_tensor, polar_surrogate):
    """ Convertit v_eff et alpha (en DEGRÉS) en forces (Fn, Ft) via la polaire d'aile PyTorch """
    alpha_rad = alpha_deg * (torch.pi / 180.0)
    r_expanded = r_tensor.expand_as(alpha_deg)
    c_expanded = c_tensor.expand_as(alpha_deg)
    
    cl_cd = polar_surrogate(alpha_deg, r_expanded)
    Cl, Cd = cl_cd[..., 0], cl_cd[..., 1]
    
    Cn = Cl * torch.cos(alpha_rad) + Cd * torch.sin(alpha_rad)
    Ct = Cl * torch.sin(alpha_rad) - Cd * torch.cos(alpha_rad)
    
    q = 0.5 * RHO * (v_eff**2) * torch.abs(c_expanded)
    return torch.stack([q * Cn, -q * Ct], dim=-1)


class PhysicsInformedLoss(nn.Module):
    """ Loss adaptée pour la prédiction cartésienne adimensionnelle (an, at) """
    def __init__(self, ae_model, scaler_v, scaler_f, lambda_val, r_tensor, c_tensor, polar_surrogate, device, v_app_train=None, v_app_val=None):
        super().__init__()
        self.ae_model = ae_model
        self.scaler_v = TorchScaler(scaler_v, device)
        self.scaler_f = TorchScaler(scaler_f, device)
        self.lambda_val = lambda_val
        self.r = r_tensor.to(device)
        self.c = c_tensor.to(device)
        self.polar_surrogate = polar_surrogate.to(device)
        self.v_app_train = v_app_train
        self.v_app_val = v_app_val
        self.mse = nn.MSELoss()

    def forward(self, z_pred, y_true_norm, v_bem_phys=None):
        y_pred_norm = self.ae_model.decode(z_pred) if self.ae_model is not None else z_pred
        
        loss_v = self.mse(y_pred_norm, y_true_norm)
        if self.lambda_val == 0.0 or self.v_app_train is None: 
            return loss_v
            
        v_pred_phys = self.scaler_v.inverse_transform(y_pred_norm)
        v_true_phys = self.scaler_v.inverse_transform(y_true_norm)
        
        if v_bem_phys is not None:
            v_pred_phys = v_pred_phys + v_bem_phys
            v_true_phys = v_true_phys + v_bem_phys
            
        is_cnn = (len(v_pred_phys.shape) == 4)
        if is_cnn:
            an_p, at_p = v_pred_phys[:, 0], v_pred_phys[:, 1]
            an_t, at_t = v_true_phys[:, 0], v_true_phys[:, 1]
        else:
            an_p, at_p = v_pred_phys[:, 0::2], v_pred_phys[:, 1::2]
            an_t, at_t = v_true_phys[:, 0::2], v_true_phys[:, 1::2]
            
        alpha_p_deg = torch.atan2(an_p, at_p) * (180.0 / torch.pi)
        alpha_t_deg = torch.atan2(an_t, at_t) * (180.0 / torch.pi)
        
        v_app_slice = self.v_app_train if self.training else self.v_app_val
        if v_app_slice is None: 
            v_app_slice = self.v_app_train
            
        v_eff_p = v_app_slice * torch.sqrt(an_p**2 + at_p**2)
        v_eff_t = v_app_slice * torch.sqrt(an_t**2 + at_t**2)
            
        f_pred_phys = convert_v_to_f_torch(v_eff_p, alpha_p_deg, self.r, self.c, self.polar_surrogate)
        f_true_phys = convert_v_to_f_torch(v_eff_t, alpha_t_deg, self.r, self.c, self.polar_surrogate)
        
        if is_cnn:
            f_pred_phys = f_pred_phys.permute(0, 3, 1, 2)
            f_true_phys = f_true_phys.permute(0, 3, 1, 2)
        else:
            f_pred_phys = f_pred_phys.reshape(v_pred_phys.shape[0], -1)
            f_true_phys = f_true_phys.reshape(v_true_phys.shape[0], -1)
            
        if is_cnn:
            D_phys = 0.5 * RHO * (v_app_slice**2) * torch.abs(self.c)
            D_phys = D_phys.unsqueeze(1) # Image (B, 1, 36, 72) pour broadcaster sur les 2 canaux Fn, Ft
        else:
            D_phys = 0.5 * RHO * (v_app_slice**2) * torch.abs(self.c)
            D_phys = torch.repeat_interleave(D_phys, 2, dim=1) # Vectoriel (B, 5184) alterné pour Fn, Ft
            
        # On divise par D avant d'appliquer la standardisation statistique
        f_pred_norm = self.scaler_f.transform(f_pred_phys / D_phys)
        f_true_norm = self.scaler_f.transform(f_true_phys / D_phys)
        # =====================================================================
        
        loss_f = self.mse(f_pred_norm, f_true_norm)
        return (1 - self.lambda_val) * loss_v + self.lambda_val * loss_f