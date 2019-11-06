from .ddpg_agent import DDPGAgent
from .critic import Critic
from .utils import soft_update

class TD3Agent(DDPGAgent):
    
    name = 'TD3'
    
    def __init__(self, config):
        super().__init__(config)
        
        self.twin_local = Critic(config.state_size * config.num_agents, 
                                 config.action_size * config.num_agents, 
                                 config.hidden_critic, 
                                 config.activ_critic)
        
        self.twin_target = Critic(config.state_size * config.num_agents, 
                                  config.action_size * config.num_agents, 
                                  config.hidden_critic, 
                                  config.activ_critic)
        
        soft_update(self.twin_local, self.twin_target, 1.0)
        
        self.twin_optim = config.optim_critic(self.twin_local.parameters(), 
                                              lr=config.lr_critic)
    
    def update_critic(self, 
                      states, 
                      actions
                      next_states, 
                      next_actions, 
                      rewards, 
                      dones):
        
        grad_clip_critic = self.config.grad_clip_critic
        use_huber_loss = self.config.use_huber_loss
        gamma = self.config.gamma
        
        # next_states => tensor(batch_size, 48)
        # next_actions => tensorbatch_size, 4)
        # Q_targets_next1 => tensor(batch_size, 1)
        Q_targets_next1 = \
            self.critic_target(next_states, next_actions).detach()
        
        Q_targets_next2 = \
            self.twin_target(next_states, next_actions).detach()
        
        Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
        
        # Compute Q targets for current states
        # tensor(batch_size, 1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # states.view(batch_size, -1) => tensor(batch_size, 48)
        # actions.view(batch_size, -1) => tensorbatch_size, 4)
        # tensor(batch_size 1)
        Q_expected1 = self.critic_local(states, actions)
        
        Q_expected2 = self.twin_local(states, actions)
        
        # Compute critic loss
        if use_huber_loss:
            critic_loss = F.smooth_l1_loss(Q_expected1, Q_targets)
            twin_loss = F.smooth_l1_loss(Q_expected2, Q_targets)
        else:
            critic_loss = F.mse_loss(Q_expected1, Q_targets)
            twin_loss = F.mse_loss(Q_expected2, Q_targets)
        
        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        
        if grad_clip_critic is not None:
            clip_grad_norm_(self.critic_local.parameters(), grad_clip_critic)
            
        self.critic_optim.step()
        
        self.twin_optim.zero_grad()
        twin_loss.backward()
        
        if grad_clip_critic is not None:
            clip_grad_norm_(self.twin_local.parameters(), grad_clip_critic)
            
        self.twin_optim.step()
    
    def summary(self, agent_name='TD3 Agent'):
        super().summary(agent_name)
        print('')
        print('Twin Network:')
        print('-------------')
        print(self.twin_local)