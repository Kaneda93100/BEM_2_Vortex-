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

class PowerLoss(nn.Module) :
    def __init__():super().__init__()

    def forward(input, target)
    