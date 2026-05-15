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
        # 1. Padding circulaire sur l'axe W (theta)
        x = F.pad(x, (self.pad_theta, self.pad_theta, 0, 0), mode='circular')
        # 2. Padding classique sur l'axe H (r)
        x = F.pad(x, (0, 0, self.pad_r, self.pad_r), mode='replicate')
        return x


class TurbineCNN(nn.Module):
    """ Stratégie GM : Réseau Entièrement Convolutif (Image-to-Image) """
    def __init__(self, in_channels=3, out_channels=2, n_layers=4, base_filters=32, dropout_rate=0.1, device='cpu'):
        super(TurbineCNN, self).__init__()
        
        layers = []
        current_in = in_channels
        
        # Empilement de couches Convolutives (Sans réduire la taille de l'image 36x36)
        for _ in range(n_layers):
            layers.append(PeriodicPadding2d(pad_r=1, pad_theta=1))
            layers.append(nn.Conv2d(current_in, base_filters, kernel_size=3, padding=0, device=device))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(base_filters, device=device))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            current_in = base_filters
            
        # Couche finale pour ramener aux 2 canaux (Fn, Ft)
        layers.append(PeriodicPadding2d(pad_r=1, pad_theta=1))
        layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=0, device=device))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x est de la forme (Batch, 3, 36, 36), la sortie sera (Batch, 2, 36, 36)
        return self.network(x)