import torch
import numpy as np

from .multi_agent import MultiAgent
from .sac_agent import SACAgent
from .utils import from_experience
from .device import device

class MultiSacAgent(MultiAgent):
    def __init__(self, config):
        super().__init__(SACAgent, config)

    def learn(self, experiences):
        
        policy_freq_update = self.config.policy_freq_update
        batch_size = self.config.batch_size
        
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        next_actions = []
        next_logs_pi = []
        for for i, agent in enumerate(self.agents):
            action, log_pi, _ = agent.policy.sample(next_states[:, i, :]
            next_actions.append(action)
            next_logs_pi.append(log_pi)
        
        next_actions = torch.cat(next_actions, dim=1).to(device)
        next_logs_pi = torch.cat(next_logs_pi, dim=1).to(device)
        
        for i, agent in enumerate(self.agents):
            agent.update_Q(states.view(batch_size, -1), 
                           actions.view(batch_size, -1), 
                           next_states.view(batch_size, -1), 
                           next_actions, 
                           next_logs_pi, 
                           rewards[:, i].view(-1, 1), 
                           dones[:, i].view(-1, 1))
        
        for agent_idx, learning_agent in enumerate(self.agents):
            
            pred_actions = []
            pred_logs_pi = []
            for i, agent in enumerate(self.agents):
                action, log_pi, _ = agent.policy.sample(states[:, i, :])
                if i != agent_idx:
                    action = action.detach()
                    log_pi = log_pi.detach()
                
                pred_actions.append(action)
                pred_logs_pi.append(log_pi)
        
            pred_actions = torch.cat(pred_actions, dim=1).to(device)
            pred_logs_pi = torch.cat(pred_logs_pi, dim=1).to(device)
        
            learning_agent.update_policy(states.view(batch_size, -1), 
                                         pred_actions, 
                                         pred_logs_pi)