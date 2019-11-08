import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal

from .gaussian_policy import GaussianPolicy
from .critic import Critic
from .utils import soft_update
from .device import device

class SACAgent:
    
    name = 'SAC'
    
    def __init__(self, config):
        
        self.policy = GaussianPolicy(config.state_size, 
                                     config.action_size, 
                                     config.hidden_actor, 
                                     config.activ_actor)
        
        self.Q1_local = Critic(config.state_size * config.num_agents, 
                               config.action_size * config.num_agents, 
                               config.hidden_critic, 
                               config.activ_critic)
        
        self.Q1_target = Critic(config.state_size * config.num_agents, 
                                config.action_size * config.num_agents, 
                                config.hidden_critic, 
                                config.activ_critic)
        
        soft_update(self.Q1_local, self.Q1_target, 1.0)
        
        self.Q1_optim = config.optim_critic(self.Q1_local.parameters(), 
                                            lr=config.lr_critic)
        
        self.Q2_local = Critic(config.state_size * config.num_agents, 
                               config.action_size * config.num_agents, 
                               config.hidden_critic, 
                               config.activ_critic)
        
        self.Q2_target = Critic(config.state_size * config.num_agents, 
                                config.action_size * config.num_agents, 
                                config.hidden_critic, 
                                config.activ_critic)
        
        soft_update(self.Q2_local, self.Q2_target, 1.0)
        
        self.Q2_optim = config.optim_critic(self.Q2_local.parameters(), 
                                            lr=config.lr_critic)
        
        # temperature variable to be learned, and its target entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.detach().exp()
        self.target_entropy = -config.action_size
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        mean, log_std = self.policy(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        # Reparameterization trick (mean + std * N(0,1))
        trick = normal.rsample()
        
        # Squashing result
        action = torch.tanh(trick)
        