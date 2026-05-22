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
    - Axe r (Hauteur/H) : Classique (Le bout de pale et le moyeu n'ont pas de voisins au-delà)
    """
    def __init__(self, pad_r, pad_theta):
        super().__init__()
        self.pad_r = pad_r
        self.pad_theta = pad_theta

    def forward(self, x):
        x = F.pad(x, (self.pad_theta, self.pad_theta, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad_r, self.pad_r), mode='replicate')
        return x


class TurbineCNN(nn.Module):
    """ 
    Stratégie GM Hybride :
    Si use_autoencoder=True : Extrait un vecteur latent condensé (Z).
    Si use_autoencoder=False : Agit comme une fonction identité (prédit directement l'image 36x72).
    """
    def __init__(self, in_channels=4, out_channels=2, use_autoencoder=False, latent_dim=64, n_layers=4, base_filters=32, dropout_rate=0.1, device='cpu'):
        super(TurbineCNN, self).__init__()
        self.use_autoencoder = use_autoencoder
        
        layers = []
        current_in = in_channels
        
        # 1. TRONC CONVOLUTIF (Maintient la taille spatiale 36x72)
        for _ in range(n_layers):
            layers.append(PeriodicPadding2d(pad_r=1, pad_theta=1))
            layers.append(nn.Conv2d(current_in, base_filters, kernel_size=3, padding=0, device=device))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(base_filters, device=device))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            current_in = base_filters
            
        self.encoder_features = nn.Sequential(*layers)
        
        # 2. TÊTE DE PRÉDICTION SÉLECTIVE
        if self.use_autoencoder:
            # Mode "Encodeur" : Écrase l'image pour prédire directement le vecteur latent Z
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_filters * 36 * 72, 256, device=device),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(256, latent_dim, device=device)
            )
        else:
            # Mode "Identité/Direct" : Termine par une convolution vers out_channels (2)
            self.head = nn.Sequential(
                PeriodicPadding2d(pad_r=1, pad_theta=1),
                nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=0, device=device)
            )
        
    def forward(self, x):
        features = self.encoder_features(x)
        return self.head(features)


class ConvolutionalAutoencoder(nn.Module):
    """ Auto-encodeur Convolutif pour le champ géométrique 36x72 """
    def __init__(self, in_channels=2, latent_dim=64, device='cpu'):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            PeriodicPadding2d(pad_r=1, pad_theta=1),
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.BatchNorm2d(16, device=device),
            
            PeriodicPadding2d(pad_r=1, pad_theta=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.BatchNorm2d(32, device=device),
            
            nn.Flatten(),
            nn.Linear(32 * 9 * 18, latent_dim, device=device)
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 32 * 9 * 18, device=device),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            PeriodicPadding2d(pad_r=1, pad_theta=1),
            nn.Conv2d(32, 16, kernel_size=3, padding=0, device=device),
            nn.ReLU(),
            nn.BatchNorm2d(16, device=device),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            PeriodicPadding2d(pad_r=1, pad_theta=1),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=0, device=device)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, 32, 9, 18)
        return self.decoder_conv(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class LinearAutoencoder(nn.Module):
    """ Auto-encodeur Dense (MLP) pour la stratégie GV (Vecteur plat de taille 5184) """
    def __init__(self, in_features=5184, latent_dim=64, device='cpu'):
        super(LinearAutoencoder, self).__init__()
        
        # Compression progressive
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(512, device=device),
            nn.Linear(512, latent_dim, device=device)
        )
        
        # Décompression progressive
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
        z = self.encode(x)
        return self.decode(z)