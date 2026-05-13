import torch
import torch.nn as nn

class TurbineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, dropout_rate, device = 'cpu'):
        super(TurbineMLP, self).__init__()
        
        layers = []
        in_features = input_dim
        
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_neurons, device = device))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_neurons
            
        layers.append(nn.Linear(in_features, output_dim, device = device))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EnsemblePointNet(nn.Module):
    """
    Contient 1296 réseaux TurbineMLP indépendants.
    Chaque réseau est spécialisé pour un couple (r, theta) précis.
    """
    def __init__(self, num_models, input_dim, output_dim, n_layers, n_neurons, dropout_rate):
        super().__init__()
        # On crée une liste PyTorch contenant les 1296 petits réseaux
        self.models = nn.ModuleList([
            TurbineMLP(input_dim, output_dim, n_layers, n_neurons, dropout_rate) 
            for _ in range(num_models)
        ])
        
    def forward(self, x):
        # x est un tenseur de forme (num_models, batch_size, input_dim)
        outputs = []
        for i, model in enumerate(self.models):
            outputs.append(model(x[i]))
        
        # On recombine les sorties en un seul tenseur
        return torch.stack(outputs)