import torch
import torch.nn as nn

from .device import device
from .base_network import BaseNetwork

class GaussianPolicy(BaseNetwork):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 activ, 
                 log_std_min=-20, log_std_max=2):
        super().__init__(activ)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.build_layers(dims)

        # Will depends on state
        self.log_std = nn.Linear(hidden_size, action_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))
        
        mean = self.layers[-1](x)
        log_std = self.log_std(x)
        
        log_std = torch.clamp(log_std, 
                              min=self.log_std_min, 
                              max=self.log_std_max)
        
        return mean, log_std