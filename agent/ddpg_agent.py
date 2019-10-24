import numpy as np
import torch
import torch.nn.functional as F

from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise
from .device import device
from .utils import make_experience, soft_update

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, config, shared_memory):
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
                                   config.hidden_critic, 
                                   config.activ_critic)
        
        self.critic_target = Critic(config.state_size, 
                                   config.hidden_critic, 
                                   config.activ_critic)
        
        soft_update(self.critic_local, self.critic_target, 1.0)
        
        self.critic_optim = config.optim_critic(self.critic_local.parameters(), 
                                                lr=config.lr_critic, 
                                                weight_decay=config.weight_decay)

        # Noise process
        self.noise = OUNoise(config.action_size, 
                             config.ou_mu, 
                             config.ou_theta, 
                             config.ou_sigma)
        
        # Shared replay memory
        self.memory = shared_memory
        
        self.noise_weight = config.noise_weight
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        batch_size = self.config.batch_size
        update_every = self.config.update_every
        
        # Save experience / reward
        experience = make_experience(state, 
                                     action, 
                                     reward, 
                                     next_state, 
                                     done)
        
        self.memory.add(experience)
        
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % update_every
        
        if self.t_step == 0:
    
            # Learn, if enough samples are available in memory
            if len(self.memory) > batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample() * self.noise_weight
        
        self.noise_weight *= self.noise_decay
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        
        gamma = self.config.gamma
        tau = self.config.tau
        
        states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None]))\
                .float().to(device)
        actions = torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None]))\
                .long().to(device)
        rewards = torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None]))\
                .float().to(device)
        next_states = torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None]))\
                .float().to(device)
        dones = torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None])\
                .astype(np.uint8)).float().to(device)
        
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        # ----------------------- update target networks ----------------------- #
        soft_update(self.critic_local, self.critic_target, tau)
        soft_update(self.actor_local, self.actor_target, tau)                     