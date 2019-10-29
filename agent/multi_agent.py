import time
import random
import torch
import numpy as np

from .replay_buffer import ReplayBuffer
from .ddpg_agent import DDPGAgent
from .utils import get_time_elapsed, make_experience

class Multi_Agent():
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
                experiences = self.memory.sample()
                for i, agent in enumerate(self.agents):
                    agent.learn(experiences, i)

    def summary(self):
        for agent in self.agents:
            agent.summary()
            print('')
    
    def train(self):
        num_episodes = self.config.num_episodes
        env_solved = self.config.env_solved
        env = self.config.env
        
        start = time.time()
        
        scores = []
        best_score = -np.inf
        
        for i_episode in range(1, num_episodes+1):
            states = self.reset()
            score = 0
            while True:
                actions = self.act(states)
                next_states, rewards, dones, _ = env.step(actions)
                
                rewards = np.array(rewards)
                dones = np.array(dones)
                
                self.step(states, actions, rewards, next_states, dones)
                
                states = next_states
                score += rewards.mean()
                
                if dones.any():
                    break
            
            scores.append(score)
            
            print('\rEpisode {}\tScore: {:.3f}'\
                  .format(i_episode, score), end='')
            
            if score > best_score:
                best_score = score
                print('\nBest score so far: {:.3f}'.format(best_score))
                
                for i, agent in enumerate(self.agents):
                    torch.save(agent.actor_local.state_dict(), 'ddpg_actor{}_checkpoint.ph'.format(i))
                
            if score >= env_solved:
                time_elapsed = get_time_elapsed(start)
                
                print('Environment solved!')
                print('Score: {:.3f}'.format(score))
                print('Time elapsed: {}'.format(time_elapsed))
                break;
                    
        env.close()
        
        return scores