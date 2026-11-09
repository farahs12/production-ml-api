import torch
from torch import nn 

class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(n_features,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear (64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    
    def forward(self,x):
        return self.layers(x)

