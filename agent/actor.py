import torch
import torch.nn as nn

from .device import device
from .utils import hidden_init

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activ):
        super().__init__()
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
        
        self.activ = activ
        
        self.to(device)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers[0:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))
        
        return torch.tanh(self.layers[-1](x)) # (-1, 1)