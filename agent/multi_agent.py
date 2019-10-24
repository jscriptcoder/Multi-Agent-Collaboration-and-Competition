from .replay_buffer import ReplayBuffer
from .ddpg_agent import DDPGAgent

class Multi_Agent():
    def __init__(self, config):
        shared_memory = ReplayBuffer(config.buffer_size, config.batch_size)
        self.agents = [DDPGAgent(config, shared_memory) for _ in range(config.num_agents)]
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    def act(self, states, add_noise=True):
        return [agent.act(state, add_noise) \
                for agent, state \
                in zip(self.agents, states)]
    
    def step(self, states, actions, rewards, next_states, dones):
        for agent, state, action, reward, next_state, done in \
        zip(self.agents, states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done)

    def train(self):
        num_episodes = self.config.num_episodes
        env_solved = self.config.env_solved
        envs = self.config.envs
        
        start = time.time()
        
        scores = []
        best_score = -np.inf
        
        for i_episode in range(1, num_episodes+1):
            self.reset()
            while True:
                policy_loss, value_loss = self.step()
                if self.done:
                    break
            
            score = self.eval_episode(1) # we evaluate only once
            scores.append(score)
            
            print('\rEpisode {}\tPolicy loss: {:.3f}\tValue loss: {:.3f}\tAvg Score: {:.3f}'\
                  .format(i_episode, 
                          policy_loss, 
                          value_loss, 
                          score), end='')
            
            if score > best_score:
                best_score = score
                print('\nBest score so far: {:.3f}'.format(best_score))
                
                torch.save(self.policy.state_dict(), '{}_actor_checkpoint.ph'.format(self.name))
                torch.save(self.value.state_dict(), '{}_critic_checkpoint.ph'.format(self.name))
                
            if score >= env_solved:
                # For speed reasons I'm gonna do a full evaluation after the env has been
                # solved the first time.
                
                print('\nRunning full evaluation...')
                
                # We now evaluate times_solved-1 (it's been already solved once), 
                # since the condition to consider the env solved is to reach the target 
                # reward at least an average of times_solved times consecutively
                avg_score = self.eval_episode(times_solved-1)
                
                if avg_score >= env_solved:
                    time_elapsed = get_time_elapsed(start)
                    
                    print('Environment solved {} times consecutively!'.format(times_solved))
                    print('Avg score: {:.3f}'.format(avg_score))
                    print('Time elapsed: {}'.format(time_elapsed))
                    break;
                else:
                    print('No success. Avg score: {:.3f}'.format(avg_score))
        
        envs.close()
        
        return scores