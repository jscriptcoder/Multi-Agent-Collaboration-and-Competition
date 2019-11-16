# Report: Tennis environment using Multi-Agent approach

## Learning Algorithm

What we're dealing with here is an envirornment with continuous observation space that consists of 8 variables corresponding to the position and velocity of the ball and racket, and continues action space, a vector with 2 numbers, corresponding to movement toward (or away from) the net, and jumping.

I'll try to solve this environment with three state-of-the-art DRL algorithms, using the latest Actor-Critic methods:

- Deep Deterministic Policy Gradient (a.k.a **DDPG**). [Paper](https://arxiv.org/abs/1509.02971)
- Twin Delayed DDPG or **TD3**. [Paper](https://arxiv.org/abs/1802.09477)
- Soft Actor Critic or **SAC**. [Paper](https://arxiv.org/abs/1801.01290). Improving on this one, there is a [second paper](https://arxiv.org/abs/1812.05905)

### Hyperparameters

Following is a list of all the hyperparameters used and their values:

#### General params
- ```seed = 0```
- ```num_agents = 2```
- ```buffer_size = int(1e6)```
- ```batch_size = 128```
- ```num_episodes = 2000```
- ```log_every = 100```
- ```num_updates = 1```, how many updates we want to perform in one learning step
- ```max_steps = 2000```, max steps done per episode if ```done``` is never ```True```
- ```gamma = 0.99```, discount factor
- ```grad_clip_actor = None```, gradient clipping for actor network
- ```grad_clip_critic = None```, gradient clipping for critic network
- ```use_huber_loss = False```, whether to use huber loss (```True```) or mse loss (```False```)
- ```update_every = 1```, how many steps before updating networks

#### General noise params
- ```use_ou_noise = True```, whether to use OU (```True```) or Gaussian (```False```) noise
- ```expl_noise = 0.1```, exploration noise in case of using Gaussian 
- ```noise_weight = 1.0```
- ```decay_noise = False```
- ```noise_linear_decay = 1e-6```
- ```noise_decay = 0.99```
- ```use_linear_decay = False```, noise_weight - noise_linear_decay (```True```), noise_weight * noise_decay (```False```)

#### Ornstein–Uhlenbeck params
- ```ou_mu = 0.0```
- ```ou_theta = 0.15```
- ```ou_sigma = 0.2```

When we reach env_solved avarage score (our target score for this environment), we'll run a full evaluation, that means, we're gonna evaluate times_solved times (this is required to solve the env) and avarage all the rewards:
- ```env_solved = 0.5```
- ```times_solved = 100```

#### DDPG params
- ```tau = 1e-2```, used in polyak averaging (soft update)
- ```lr_actor = 1e-3```
- ```lr_critic = 1e-3```
- ```hidden_actor = (64, 64)```
- ```hidden_critic = (512,)```
- ```activ_actor = F.relu```
- ```activ_critic = F.relu```
- ```optim_actor = Adam```
- ```optim_critic = Adam```

#### TD3 params
- ```tau = 1e-2```,
- ```lr_actor = 1e-3```
- ```lr_critic = 1e-3```
- ```hidden_actor = (64, 64)```
- ```hidden_critic = (256, 256)```
- ```activ_actor = F.relu```
- ```activ_critic = F.relu```
- ```optim_actor = Adam```
- ```optim_critic = Adam```
- ```policy_noise = 0.1```, target policy smoothing by adding noise to the target action
- ```noise_clip = 0.1```, clipping value for the noise added to the target action
- ```policy_freq_update = 2```, how many critic net updates before updating the actor

#### SAC params
- ```tau = 1e-2```,
- ```lr_actor = 1e-3```
- ```lr_critic = 1e-3```
- ```hidden_actor = (64, 64)```
- ```hidden_critic = (512,)```
- ```activ_actor = F.relu```
- ```activ_critic = F.relu```
- ```optim_actor = Adam```
- ```optim_critic = Adam```
- ```log_std_min=-20```, min value of the log std calculated by the Gaussian policy
- ```log_std_max=2```, max value of the log std calculated by the Gaussian policy
- ```alpha = 0.01```,
- ```alpha_auto_tuning = True```, when ```True```, ```alpha``` is a learnable
- ```optim_alpha = Adam```, optimizer for alpha
- ```lr_alpha = 3e-4```, learning rate for alpha

### Algorithms
**Deep Deterministic Policy Gradient or DDPG**:

TODO

**Twin Delayed DDPG or TD3**:

TODO

**Soft Actor Critic or SAC**:

TODO

### Neural Networks Architecture

TODO

## Plot of Rewards

1. **DDPG**:
TODO

2. **TD3**:
TODO

## Ideas for Future Work
TODO
