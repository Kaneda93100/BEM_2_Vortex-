import torch
from torch import nn
import json

class NNsin(nn.Module) :
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
    


if __name__ == '__main__':
    feat = torch.linspace(0, 2*torch.pi, 100)
    label = torch.sin(feat)