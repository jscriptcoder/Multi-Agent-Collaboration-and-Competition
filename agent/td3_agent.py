from .ddpg_agent import DDPGAgent
from .critic import Critic
from .utils import soft_update

class TD3Agent(DDPGAgent):
    
    name = 'TD3'
    
    def __init__(self, config):
        super().__init__(config)
        
#        self.twin_local = Critic(config.state_size * config.num_agents, 
#                                 config.action_size * config.num_agents, 
#                                 config.hidden_critic, 
#                                 config.activ_critic)
#        
#        self.twin_target = Critic(config.state_size * config.num_agents, 
#                                  config.action_size * config.num_agents, 
#                                  config.hidden_critic, 
#                                  config.activ_critic)
#        
#        soft_update(self.twin_local, self.twin_target, 1.0)
#        
#        self.twin_optim = config.optim_critic(self.twin_local.parameters(), 
#                                              lr=config.lr_critic)
    
    def summary(self, agent_name='TD3 Agent'):
        super().summary(agent_name)
        print('')
        print('Twin Network:')
        print('-------------')
#        print(self.twin_local)