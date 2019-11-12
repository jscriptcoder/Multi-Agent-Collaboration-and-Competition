import time
import random
import torch
import numpy as np
from collections import deque

from .replay_buffer import ReplayBuffer
from .utils import make_experience, from_experience
from .utils import get_time_elapsed
from .device import device

class MultiAgent():
    def __init__(self, Agent, config):
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.config = config

        self.memory = ReplayBuffer(config.buffer_size, config.batch_size)
        
        self.agents = [Agent(config) for _ in range(config.num_agents)]
        
        self.t_step = 0
        self.p_update = 0
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
        
        return self.config.env.reset()
    
    def act(self, states, train=True):
        return np.array([agent.act(state, train) \
                         for agent, state \
                         in zip(self.agents, states)])
    
    def step(self, states, actions, rewards, next_states, dones):
        batch_size = self.config.batch_size
        update_every = self.config.update_every
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
                
                # Multiple updates in one learning step
                for _ in range(num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def learn(self, experiences):
        
        policy_freq_update = self.config.policy_freq_update
        batch_size = self.config.batch_size
        
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        # Collect the actions for all the agents in the next state
        next_actions = [agent.actor_target(next_states[:, i, :]) \
                        for i, agent in enumerate(self.agents)]
        
        # tensor(batch_size, 4)
        next_actions = torch.cat(next_actions, dim=1).to(device)
        
        for i, agent in enumerate(self.agents):
            agent.update_critic(states.view(batch_size, -1), 
                                actions.view(batch_size, -1), 
                                next_states.view(batch_size, -1), 
                                next_actions, 
                                rewards[:, i].view(-1, 1), 
                                dones[:, i].view(-1, 1))
        
        self.p_update = (self.p_update + 1) % policy_freq_update
        
        if self.p_update == 0:
            # ---------------------------- update actor ---------------------------- #
            for agent_idx, learning_agent in enumerate(self.agents):
                # Collect the actions for all the agents in that state
                # We need to detach actions for all the other agents
                pred_actions = [agent.actor_local(states[:, i, :]) \
                                if i == agent_idx \
                                else agent.actor_local(states[:, i, :]).detach() \
                                for i, agent in enumerate(self.agents)]
            
                pred_actions = torch.cat(pred_actions, dim=1).to(device)
            
                learning_agent.update_actor(states.view(batch_size, -1), 
                                            pred_actions)

    def summary(self):
        for i, agent in enumerate(self.agents):
            agent.summary('{} Agent {}'.format(agent.name, i))
            print('')
    
    def train(self):
        num_episodes = self.config.num_episodes
        max_steps = self.config.max_steps
        num_agents = self.config.num_agents
        log_every = self.config.log_every
        env_solved = self.config.env_solved
        times_solved = self.config.times_solved
        env = self.config.env
        
        start = time.time()
        
        scores_window = deque(maxlen=times_solved)
        best_score = -np.inf
        scores = []

        for i_episode in range(1, num_episodes+1):
            states = self.reset()
            score = np.zeros(num_agents)
            
            for _ in range(max_steps):
                actions = self.act(states)
                next_states, rewards, dones, _ = env.step(actions)
                
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
            
            print('\rEpisode {}\tAvg Score: {:.3f}'\
                  .format(i_episode, avg_score), end='')
            
            if i_episode % log_every == 0:
                print('\rEpisode {}\tAvg Score: {:.3f}'\
                  .format(i_episode, avg_score))
            
            if avg_score > best_score:
                best_score = avg_score
                self.save_agents_weights()
                
            if avg_score >= env_solved:
                print('\nRunning evaluation without noise...')

                avg_score = self.eval_episode()

                if avg_score >= env_solved:
                    time_elapsed = get_time_elapsed(start)

                    print('Environment solved {} times consecutively!'.format(times_solved))
                    print('Avg score: {:.3f}'.format(avg_score))
                    print('Time elapsed: {}'.format(time_elapsed))
                    break;
                else:
                    print('No success. Avg score: {:.3f}'.format(avg_score))
                    
        env.close()
        
        return scores
    
    def save_agents_weights(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 
                       '{}_actor{}_checkpoint.ph'.format(agent.name, i))
    
    def eval_episode(self):
        num_agents = self.config.num_agents
        times_solved = self.config.times_solved
        env = self.config.env
        
        total_reward = np.zeros(num_agents)
        
        for _ in range(times_solved):
            states = env.reset()
            while True:
                actions = self.act(states, train=False)
                states, rewards, dones, _ = env.step(actions)

                total_reward += rewards
    
                if np.any(dones):
                    break
                
        return np.max(total_reward) / times_solved