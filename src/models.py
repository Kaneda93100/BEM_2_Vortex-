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