import torch
import torch.nn as nn

from .device import device
from .utils import hidden_init

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activ):
        super().__init__()
        
        dims = (state_size + action_size,) + hidden_size + (1,)
        
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

    def forward(self, state, action):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        if type(action) != torch.Tensor:
            state = torch.FloatTensor(action).to(device)
        
        x = torch.cat((state, action.float()), dim=1)
        
        for layer in self.layers[:-1]:
            x = self.activ(layer(x))
    
        return self.layers[-1](x)