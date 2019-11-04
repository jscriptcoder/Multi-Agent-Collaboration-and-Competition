import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from collections import deque

from .replay_buffer import ReplayBuffer
from .utils import make_experience, from_experience
from .utils import soft_update, get_time_elapsed
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
        self.use_twin = hasattr(self.agents[0],'twin_local')
        
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
        
        return self.config.env.reset()
    
    def act(self, states, add_noise=True):
        return np.array([agent.act(state, add_noise=add_noise) \
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
                
                # Multiple updates in one learning step
                for _ in range(num_updates):
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
        policy_freq_update = self.config.policy_freq_update
        policy_noise = self.config.policy_noise
        noise_clip = self.config.noise_clip
        grad_clip_actor = self.config.grad_clip_actor
        grad_clip_critic = self.config.grad_clip_critic
        batch_size = self.config.batch_size
        use_huber_loss = self.config.use_huber_loss
        gamma = self.config.gamma
        tau = self.config.tau
        
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        # Current agent learning
        learning_agent = self.agents[agent_idx]
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        # Target Policy Smoothing Regularization: add a small amount of clipped 
        # random noises to the selected action
        if self.use_twin and policy_noise > 0.0:
            noise = torch.empty_like(actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            
            actions_next = [(agent.actor_target(next_states[:, i, :]) + \
                             noise[:, i, :]).clamp(-1., 1.) \
                            for i, agent in enumerate(self.agents)]
        else:
            # Collect the actions for all the agents in the next state
            actions_next = [agent.actor_target(next_states[:, i, :]) \
                            for i, agent in enumerate(self.agents)]
        
        # tensor(batch_size, 4)
        actions_next = torch.cat(actions_next, dim=1).to(device)
        
        # next_states.view(batch_size, -1) => tensor(batch_size, 48)
        # tensor(batch_size, 1)
        Q_targets_next = \
            learning_agent.critic_target(next_states.view(batch_size, -1), 
                                         actions_next).detach()
        
        if self.use_twin:
            Q_targets_next2 = \
                learning_agent.twin_target(next_states.view(batch_size, -1), 
                                           actions_next).detach()
            
            Q_targets_next = torch.min(Q_targets_next, Q_targets_next2)
        
        # tensor(batch_size, 1)
        rewards = rewards[:, agent_idx].view(-1, 1)
        
        # tensor(batch_size, 1)
        dones = dones[:, agent_idx].view(-1, 1)

        # Compute Q targets for current states
        # tensor(batch_size, 1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # states.view(batch_size, -1) => tensor(batch_size, 48)
        # actions.view(batch_size, -1) => tensorbatch_size, 4)
        # tensor(batch_size 1)
        Q_expected = \
            learning_agent.critic_local(states.view(batch_size, -1), 
                                        actions.view(batch_size, -1))
        
        
        # Compute critic loss
        if use_huber_loss:
            critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        learning_agent.critic_optim.zero_grad()
        critic_loss.backward()
        
        if grad_clip_critic is not None:
            clip_grad_norm_(learning_agent.critic_local.parameters(), 
                            grad_clip_critic)
            
        learning_agent.critic_optim.step()
        
        # Compute critic loss twin Critic network
        if self.use_twin:
            Q_expected2 = \
                learning_agent.twin_local(states.view(batch_size, -1), 
                                          actions.view(batch_size, -1))
            
            if use_huber_loss:
                twin_loss = F.smooth_l1_loss(Q_expected2, Q_targets)
            else:
                twin_loss = F.mse_loss(Q_expected2, Q_targets)
            
            # Minimize the loss
            learning_agent.twin_optim.zero_grad()
            twin_loss.backward()
            
            if grad_clip_critic is not None:
                clip_grad_norm_(learning_agent.twin_local.parameters(), 
                                grad_clip_critic)
                
            learning_agent.twin_optim.step()
        
        self.p_update = (self.p_update + 1) % policy_freq_update
        
        if self.p_update == 0:
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
            
            if grad_clip_actor is not None:
                clip_grad_norm_(learning_agent.actor_local.parameters(), 
                                grad_clip_actor)
                
            learning_agent.actor_optim.step()
            
            # ----------------------- update target networks ----------------------- #
            soft_update(learning_agent.critic_local, 
                        learning_agent.critic_target, 
                        tau)
            
            if self.use_twin:
                soft_update(learning_agent.twin_local, 
                            learning_agent.twin_target, 
                            tau)
            
            soft_update(learning_agent.actor_local, 
                        learning_agent.actor_target, 
                        tau) 

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
                
                for i, agent in enumerate(self.agents):
                    torch.save(agent.actor_local.state_dict(), 'ddpg_actor{}_checkpoint.ph'.format(i))
                
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
    
    def eval_episode(self):
        """Evaluation method. 
        Will run times_solved times and avarage the total reward obtained
        """
        
        num_agents = self.config.num_agents
        times_solved = self.config.times_solved
        env = self.config.env
        
        total_reward = np.zeros(num_agents)
        
        for _ in range(times_solved):
            states = env.reset()
            while True:
                actions = self.act(states, add_noise=False)
                states, rewards, dones, _ = env.step(actions)

                total_reward += rewards
    
                if np.any(dones):
                    break
                
        return np.max(total_reward) / times_solved