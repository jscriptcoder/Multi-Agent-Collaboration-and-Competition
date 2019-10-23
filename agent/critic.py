import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import device

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, activ):
        super().__init__()
        
        dims = (state_size,) + hidden_size + (1,)
        
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
        
        self.activ = activ
        
        self.to(device)

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))

        value = self.layers[-1](x)
    
        return value