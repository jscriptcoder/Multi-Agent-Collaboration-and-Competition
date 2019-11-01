import numpy as np

from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    name = 'DDPG'
    
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
        
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        
        self.actor_optim = config.optim_actor(self.actor_local.parameters(), 
                                              lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(config.state_size * config.num_agents,
                                   config.action_size * config.num_agents,
                                   config.hidden_critic, 
                                   config.activ_critic)
        
        self.critic_target = Critic(config.state_size * config.num_agents, 
                                    config.action_size * config.num_agents,
                                    config.hidden_critic, 
                                    config.activ_critic)
        
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        
        self.critic_optim = config.optim_critic(self.critic_local.parameters(), 
                                                lr=config.lr_critic)

        # Noise process
        # TODO: how about trying simple gaussian noise?:
        #       np.random.normal(0, expl_noise, size=action_size)
        self.noise = OUNoise(config.action_size, 
                             config.ou_mu, 
                             config.ou_theta, 
                             config.ou_sigma)
        
        self.noise_weight = config.noise_weight
    
    def act(self, state, add_noise=True, decay_noise=False):
        """Returns actions for given state as per current policy."""
        
        noise_decay = self.config.noise_decay
        use_linear_decay = self.config.use_linear_decay
        noise_linear_decay = self.config.noise_linear_decay
        
        action = self.actor_local(state).cpu().data.numpy()
        
        if add_noise:
            action += self.noise.sample() * self.noise_weight
            
            if decay_noise:
                if use_linear_decay:
                    self.noise_weight = max(0.1, 
                                            self.noise_weight - noise_linear_decay)
                else:
                    self.noise_weight = max(0.1, 
                                            self.noise_weight * noise_decay)
        
        return np.clip(action, -1, 1)

    def reset(self):
        if not self.config.noisy_net:
            self.noise.reset()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def summary(self, agent_name='DDGP Agent'):
        print('{}:'.format(agent_name))
        print('==========')
        print('')
        print('Actor Network:')
        print('--------------')
        print(self.actor_local)  
        print('')
        print('Critic Network:')
        print('---------------')
        print(self.critic_local)
        
           