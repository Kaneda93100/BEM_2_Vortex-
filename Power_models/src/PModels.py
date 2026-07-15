import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.config import RHO, KERNEL_SIZE, PADDING_R, PADDING_THETA 
from core.physics import compute_density_diff
from core.models import TorchScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerMLP(nn.Module) :
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, dropout, act_func = nn.ReLU, device = 'cpu') :
        super().__init__()
        
        layers = []
        in_features = input_dim

        for _ in range(n_layers) :
            layers.append(nn.Linear(in_features, n_neurons, device = device))
            layers.append(act_func())
            layers.append(nn.Dropout(dropout))
            in_features = n_neurons

        layers.append(nn.Linear(in_features, output_dim, device = device))
        self.network = nn.Sequential(*layers)

    def forward(self, x) :
        return self.network(x)
class ForceEncoder(nn.Module) :
    def __init__(self, in_features = 5184, latent_dim = 64, device = 'cpu') :
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512, device = device),
            nn.ReLU(),
            nn.BatchNorm1d(512, device = device),
            nn.Linear(512, latent_dim, device = device)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512, device = device),
            nn.ReLU(),
            nn.BatchNorm1d(512, device = device),
            nn.Linear(512, in_features, device = device)
        )
        
    def encode(self,x) :
        return self.encoder(x)
    def decode(self, x) :
        return self.decoder(x)
    def forward(self, x) :
        return self.decode(self.encode(x))

class PowerCNN(nn.Module) : 
    def __init__(self, in_channels, n_layers, base_filters, dropout_rate, size_output = 2592, device = 'cpu') :
        super().__init__()
        
        ## Entrée du réseau
        self.initial_layer = nn.Sequential(
            PeriodicPadding2d(pad_r = PADDING_R, pad_theta = PADDING_THETA), ## Padding sur l'image avant convolution pour capter le bord
            nn.Conv2d(in_channels, base_filters, kernel_size = KERNEL_SIZE, padding = 0, device = device), ## Duplication de l'image sur base_filters channel différent sur lesquels un filtres particulier est appliqué
            nn.BatchNorm2d(base_filters, device = device), ## Normalisation sur tout le batch pour chacun des channels, si on est en full batch, alors chaque channel est normalisé indépendamment des autres
            nn.ReLU() ## Passage dans une fonction d'activation : chaque pixel de chaque channel y est passé
        )

        ## Création de chacune des couches cachés
        layers = []
        curr_filters = base_filters
        for _ in range(n_layers) :
            layers.append(ResBlockPeriodic(curr_filters, curr_filters, device = device))
            if dropout_rate > 0 :
                layers.append(nn.Dropout2d(dropout_rate))
        
        self.hidden_layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(
            PeriodicPadding2d(pad_r = PADDING_R, pad_theta = PADDING_THETA),
            nn.Conv2d(curr_filters, out_channels = 1, kernel_size = KERNEL_SIZE, padding = 0, device = device), 
            nn.Flatten(start_dim = 1), 
            nn.Linear(36 * 72, size_output, device = device)
        )

    def forward(self,x) :
            x = self.initial_layer(x)
            x = self.hidden_layers(x)
            x = self.final_layer(x)
            return x

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

class PowerDensityLoss(nn.Module) :
    def __init__(self):
        super().__init__()

    ## Métrique à essayer sur les GV, pas les GVP
    def forward(self, entree, scaler, mdl_output, target, device) :

        if entree == 'GMP' :
            
            denorm_out = scaler.inverse_transform(mdl_output)
            denorm_targ = scaler.inverse_transform(target)

            denorm_out = denorm_out.reshape((denorm_out.shape[0],2,2592))
            denorm_targ = denorm_targ.reshape((denorm_targ.shape[0],2,2592,))


            ## out_dP et targ_dP sont les densités de puissances (des mesures)
            out_dP = compute_cp_diff(col_fn = denorm_out[:,0,:], col_ft = denorm_out[:,1,:], 
                                    device = device)
        
        return nn.MSELoss(out_dP, target)
    
class PowerLoss(nn.Module) : 
    def __init__(self) :
        super().__init__()

    def forward(self, scaler_field, scaler_scalar, output, target, entree, res, device):
        if entree == 'GV' :
            ## Dénormaliser
            denorm_out = scaler_field.inverse_transform(output) 
            denorm_targ = scaler_scalar.inverse_transform(target)      
            if res == '1' :
                denorm_out += denorm_targ

            ## Réajuster les dimensions et calculer les puissances moyennées
            denorm_out = denorm_out.reshape((denorm_out.shape[0],2,2592))
            out_dP = compute_cp_diff(col_fn = denorm_out[:,0,:], col_ft = denorm_out[:,1,:], 
                                    device = device)
        
        elif entree == 'GMP' :
            ## Dénormaliser
            denorm_out = scaler_field.inverse_transform(output) 
            denorm_targ = scaler_scalar.inverse_transform(target) 

            if res == '1' :
                denorm_out += denorm_targ
            
            denorm_out = denorm_out.reshape((denorm_out.shape[0],2,2592))
            out_dP = compute_cp_diff(col_fn = denorm_out[:,0,:], col_ft = denorm_out[:,1,:], 
                                    device = device)

        out_cp_mean = torch.zeros_like(denorm_targ)
        for i in range(out_dP.shape[0]) :
            out_cp_mean[i,0] = torch.mean(out_dP[i,:])
        crit = nn.MSELoss()
        ## Prêt à rétro-propager !
        return crit(out_cp_mean, denorm_targ)
    