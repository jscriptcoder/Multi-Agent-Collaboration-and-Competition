import torch

from .multi_agent import MultiAgent
from .sac_agent import SACAgent
from .utils import from_experience
from .device import device

class MultiSacAgent(MultiAgent):
    def __init__(self, config):
        super().__init__(SACAgent, config)

    def learn(self, experiences):
        batch_size = self.config.batch_size
        
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        next_actions = []
        next_log_probs = []
        for i, agent in enumerate(self.agents):
            action, log_prob, _ = agent.policy.sample(next_states[:, i, :])
            next_actions.append(action)
            next_log_probs.append(log_prob)
        
        next_actions = torch.cat(next_actions, dim=1).to(device)
        next_log_probs = torch.cat(next_log_probs, dim=1).to(device)
        
        print(next_actions.size())
        print(next_log_probs.size())
        raise 'stop'
        
        for agent_idx, learning_agent in enumerate(self.agents):
            # Update Q values
            learning_agent.update_Q(states.view(batch_size, -1), 
                                    actions.view(batch_size, -1), 
                                    next_states.view(batch_size, -1), 
                                    next_actions, 
                                    next_log_probs, 
                                    rewards[:, agent_idx].view(-1, 1), 
                                    dones[:, agent_idx].view(-1, 1))
            
            pred_actions = []
            pred_log_props = []
            for i, agent in enumerate(self.agents):
                action, log_prob, _ = agent.policy.sample(states[:, i, :])
                if i != agent_idx:
                    action = action.detach()
                    log_prob = log_prob.detach()
                
                pred_actions.append(action)
                pred_log_props.append(log_prob)
        
            pred_actions = torch.cat(pred_actions, dim=1).to(device)
            pred_log_props = torch.cat(pred_log_props, dim=1).to(device)
        
            # Update Policy
            learning_agent.update_policy(states.view(batch_size, -1), 
                                         pred_actions, 
                                         pred_log_props)
            
            # Update temeperature
            learning_agent.try_update_alpha(pred_log_props)
    
    def save_agents_weights(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.policy.state_dict(), 
                   '{}_actor{}_checkpoint.ph'.format(agent.name, i))