import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """ 
    Padding physique pour une grille polaire (r, theta) :
    - Axe theta (Largeur/W) : Circulaire (0° est voisin de 360°)
    - Axe r (Hauteur/H) : Classique (replicate)
    """
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
        self.pad = PeriodicPadding2d(pad_r=1, pad_theta=1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, device=device)
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
        out = self.relu(out)
        return out


class TurbineCNN(nn.Module):
    """ 
    Stratégie GM : Réseau Convolutif Profond pour prédictions spatiales (H=36, W=72)
    Supporte à la fois la prédiction directe de grilles et la prédiction de l'espace latent Z.
    """
    def __init__(self, in_channels, out_channels, use_autoencoder=False, latent_dim=64, n_layers=4, base_filters=64, dropout_rate=0.1, device='cpu'):
        super(TurbineCNN, self).__init__()
        self.use_autoencoder = use_autoencoder

        # Le tenseur d'entrée est déjà une image (ex: 4, 36, 72). 
        # On utilise une couche convolutive directe pour lire ces canaux.
        self.initial_conv = nn.Sequential(
            PeriodicPadding2d(pad_r=1, pad_theta=1),
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=0, device=device),
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
        
        # Tête adaptative en fonction de la stratégie (Avec ou sans Auto-encodeur)
        if use_autoencoder:
            self.final_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(curr_filters * 36 * 72, latent_dim, device=device)
            )
        else:
            self.final_head = nn.Sequential(
                PeriodicPadding2d(pad_r=1, pad_theta=1),
                nn.Conv2d(curr_filters, out_channels, kernel_size=3, padding=0, device=device)
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
        
        # Tableaux de suivi pour mémoriser l'évolution géométrique exacte de la grille (36x72)
        self.target_shapes = []
        h_current, w_current = 36, 72 
        
        # --- STRUCTURE DE L'ENCODEUR ---
        for i in range(depth):
            self.target_shapes.append((h_current, w_current))
            
            enc_layers.append(PeriodicPadding2d(pad_r=1, pad_theta=1))
            enc_layers.append(nn.Conv2d(curr_in, curr_f, kernel_size=3, stride=2, padding=0, device=device))
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

        # --- STRUCTURE DU DÉCODEUR ---
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
            dec_layers.append(PeriodicPadding2d(pad_r=1, pad_theta=1))
            dec_layers.append(nn.Conv2d(curr_in_dec, next_f, kernel_size=3, padding=0, device=device))
            
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
        z = self.encode(x)
        return self.decode(z)


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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decoder(self.encoder(x))