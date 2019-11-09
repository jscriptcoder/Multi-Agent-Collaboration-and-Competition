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
                                     config.activ_actor, 
                                     config.log_std_min, 
                                     config.log_std_max)
        
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
        self.alpha_optim = config.optim_alpha([self.log_alpha], lr=config.lr_alpha)
    
    def act(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        
        return action.detach().cpu().numpy()
    
    def update_policy(self, 
                      states, 
                      actions,
                      next_states, 
                      next_actions, 
                      rewards, 
                      dones):
        
#        with torch.no_grad():
        pass
        