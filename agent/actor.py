import torch

from .device import device
from .base_network import BaseNetwork

class Actor(BaseNetwork):
    def __init__(self, state_size, action_size, hidden_size, activ):
        super().__init__(activ)
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.build_layers(dims)
        
        self.reset_parameters()

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))
        
        return torch.tanh(self.layers[-1](x)) # (-1, 1)