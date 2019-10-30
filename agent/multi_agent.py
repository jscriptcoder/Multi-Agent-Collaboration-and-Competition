import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from .replay_buffer import ReplayBuffer
from .ddpg_agent import DDPGAgent
from .utils import get_time_elapsed, make_experience, soft_update
from .device import device

class MultiAgent():
    def __init__(self, config):
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.config = config

        self.memory = ReplayBuffer(config.buffer_size, config.batch_size)
        
        self.agents = [DDPGAgent(config) \
                       for _ in range(config.num_agents)]
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
        
        return self.config.env.reset()
    
    def act(self, states, add_noise=True):
        return np.array([agent.act(state, add_noise) \
                         for agent, state \
                         in zip(self.agents, states)])
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        batch_size = self.config.batch_size
        update_every = self.config.update_every
        num_agents = self.config.num_agents
        num_updates = self.config.num_updates
        
        experience = make_experience(states, 
                                     actions, 
                                     rewards, 
                                     next_states, 
                                     dones)
        self.memory.add(experience)
        
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % update_every
        
        if self.t_step == 0:
    
            # Learn, if enough samples are available in memory
            if len(self.memory) > batch_size:
                
                for update in range(num_updates):
                    experiences = self.memory.sample()
                    for i in range(num_agents):
                        self.learn(experiences, i)

    def learn(self, experiences, agent_idx):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        batch_size = self.config.batch_size
        gamma = self.config.gamma
        tau = self.config.tau
        
        # torch.Size([batch_size, num_agents, state_size])
        states = torch.from_numpy(
                np.array([e.state for e in experiences if e is not None]))\
                .float().to(device)
        
        # torch.Size([batch_size, num_agents, action_size])
        actions = torch.from_numpy(
                np.array([e.action for e in experiences if e is not None]))\
                .long().to(device)
        
        # torch.Size([batch_size, num_agents])
        rewards = torch.from_numpy(
                np.array([e.reward for e in experiences if e is not None]))\
                .float().to(device)
        
        # torch.Size([batch_size, num_agents, state_size])
        next_states = torch.from_numpy(
                np.array([e.next_state for e in experiences if e is not None]))\
                .float().to(device)
        
        # torch.Size([batch_size, num_agents])
        dones = torch.from_numpy(
                np.array([e.done for e in experiences if e is not None])\
                .astype(np.uint8)).float().to(device) 
        
        # Current agent learning
        learning_agent = self.agents[agent_idx]
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        # Collect the actions for all the agents in the next state
        actions_next = [agent.actor_target(next_states[:, i, :]) \
                        for i, agent in enumerate(self.agents)]
        
        # torch.Size([batch_size, 4])
        actions_next = torch.cat(actions_next, dim=1).to(device)
        
        # next_states.view(batch_size, -1) => torch.Size([batch_size, 48])
        # torch.Size([batch_size, 1])
        Q_targets_next = \
            learning_agent.critic_target(next_states.view(batch_size, -1), 
                                         actions_next).detach()
        
        # torch.Size([batch_size, 1])
        rewards = rewards[:, agent_idx].view(-1, 1)
        
        # torch.Size([batch_size, 1])
        dones = dones[:, agent_idx].view(-1, 1)

        # Compute Q targets for current states
        # torch.Size([batch_size, 1])
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        # states.view(batch_size, -1) => torch.Size([batch_size, 48])
        # actions.view(batch_size, -1) => torch.Size([batch_size, 4])
        # torch.Size([batch_size 1])
        Q_expected = \
            learning_agent.critic_local(states.view(batch_size, -1), 
                                        actions.view(batch_size, -1))
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
#        huber_loss = torch.nn.SmoothL1Loss()
#        critic_loss = huber_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        learning_agent.critic_optim.zero_grad()
        critic_loss.backward()
        learning_agent.critic_optim.step()
        
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        
        # Collect the actions for all the agents in that state
        actions_pred = [agent.actor_local(states[:, i, :]) \
                        if i == agent_idx \
                        else agent.actor_local(states[:, i, :]).detach() \
                        for i, agent in enumerate(self.agents)]
        
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        actor_loss = \
            -learning_agent.critic_local(states.view(batch_size, -1), 
                                         actions_pred).mean()
        
        # Minimize the loss
        learning_agent.actor_optim.zero_grad()
        actor_loss.backward()
        learning_agent.actor_optim.step()
        
        
        # ----------------------- update target networks ----------------------- #
        soft_update(learning_agent.critic_local, learning_agent.critic_target, tau)
        soft_update(learning_agent.actor_local, learning_agent.actor_target, tau) 

    def summary(self):
        for agent in self.agents:
            agent.summary()
            print('')
    
    def train(self):
        scores_window = deque(maxlen=self.config.times_solved)
        num_episodes = self.config.num_episodes
        max_steps = self.config.max_steps
        num_agents = self.config.num_agents
        log_every = self.config.log_every
        env_solved = self.config.env_solved
        env = self.config.env
        
        start = time.time()
        
        scores = []
        best_score = -np.inf
        
        for i_episode in range(1, num_episodes+1):
            states = self.reset()
            score = np.zeros(num_agents)
            
            for step in range(max_steps):
                actions = self.act(states)
                next_states, rewards, dones, _ = env.step(actions.cpu().numpy())
                
                rewards = np.array(rewards)
                dones = np.array(dones)
                
                self.step(states, actions, rewards, next_states, dones)
                
                states = next_states
                score += rewards
                
                if dones.any():
                    break
            
            max_score = np.max(score)
            scores.append(max_score)
            scores_window.append(max_score)
            avg_score = np.mean(scores_window)
            
            print('\rEpisode {}\tAvg score: {:.3f}'\
                  .format(i_episode, avg_score), end='')
            
            if i_episode % log_every == 0:
                print('\rEpisode {}\tAvg score: {:.3f}'\
                  .format(i_episode, avg_score))
            
            if avg_score > best_score:
                best_score = avg_score
#                print('\nBest avg score so far: {:.3f}'.format(best_score))
                
                for i, agent in enumerate(self.agents):
                    torch.save(agent.actor_local.state_dict(), 'ddpg_actor{}_checkpoint.ph'.format(i))
                
            if avg_score >= env_solved:
                time_elapsed = get_time_elapsed(start)
                
                print('Environment solved!')
                print('Avg score: {:.3f}'.format(avg_score))
                print('Time elapsed: {}'.format(time_elapsed))
                break;
                    
        env.close()
        
        return scores