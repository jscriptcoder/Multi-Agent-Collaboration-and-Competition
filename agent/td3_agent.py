from .ddpg_agent import DDPGAgent
from .critic import Critic

class TD3Agent(DDPGAgent):
    
    name = 'TD3'
    
    def __init__(self, config):
        super().__init__(config)
        
        self.critic_local2 = Critic(config.state_size * config.num_agents,
                                    config.action_size * config.num_agents,
                                    config.hidden_critic, 
                                    config.activ_critic)
        
        self.critic_target2 = Critic(config.state_size * config.num_agents, 
                                     config.action_size * config.num_agents,
                                     config.hidden_critic, 
                                     config.activ_critic)
        
        self.soft_update(self.critic_local2, self.critic_target2, 1.0)
        
        self.critic_optim2 = config.optim_critic(self.critic_local2.parameters(), 
                                                 lr=config.lr_critic)
    
    def summary(self, agent_name='TD3 Agent'):
        super().summary(agent_name)
        print('')
        print('Twin Critic Network:')
        print('---------------')
        print(self.critic_local2)