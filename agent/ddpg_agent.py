import numpy as np
import torch

from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise
from .utils import soft_update

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, config):
        """Initialize an Agent object.
        
        Params
        ======
            config (Config)
        """
        
        self.config = config

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(config.state_size, 
                                 config.action_size, 
                                 config.hidden_actor, 
                                 config.activ_actor)
        
        self.actor_target = Actor(config.state_size, 
                                  config.action_size, 
                                  config.hidden_actor, 
                                  config.activ_actor)
        
        soft_update(self.actor_local, self.actor_target, 1.0)
        
        self.actor_optim = config.optim_actor(self.actor_local.parameters(), 
                                              lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(config.state_size,
                                   config.action_size,
                                   config.hidden_critic, 
                                   config.activ_critic, 
                                   config.num_agents)
        
        self.critic_target = Critic(config.state_size, 
                                    config.action_size,
                                    config.hidden_critic, 
                                    config.activ_critic, 
                                    config.num_agents)
        
        soft_update(self.critic_local, self.critic_target, 1.0)
        
        self.critic_optim = config.optim_critic(self.critic_local.parameters(), 
                                                lr=config.lr_critic)

        # Noise process
        self.noise = OUNoise(config.action_size, 
                             config.ou_mu, 
                             config.ou_theta, 
                             config.ou_sigma)
        
        self.noise_weight = config.noise_weight
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        noise_decay = self.config.noise_decay
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample() * self.noise_weight
        
        self.noise_weight = max(0.1, self.noise_weight * noise_decay)
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()       

    def summary(self):
        print('DDGPAgent:')
        print('==========')
        print(self.actor_local)             