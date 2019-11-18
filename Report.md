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

DDPG is an off-policy algorithm considered to be the Q-learning version for continuos action space. It uses four neural nets: a Q-network, a (deterministic) policy network and their respective target networks. The Q-function is used to learn the policy, and it does so by using off policy data and the Bellman equation. Remember, in Q-learning if we know the optimal ```Q(s,a)```, finding the optimal action in a given state is as simple as solving:

<p align="center"><img src="images/optimal_action.svg" /></p>

Finding <img src="images/max_a.svg"> when the action space is continous is not trivial. But because of this continuity in the action space, we assume that the optimal ```Q(s,a)``` is differentiable with respect to action. This makes it possible to use a gradient based learning rule for ```μ(s)``` and instead approximate it:


<p align="center"><img src="images/max_q_approx.svg" /></p>

DDPG uses two tricks to make the learning more stable:

1. _Replay Buffer_: since DDPG is an off policy algorithm, it can make use of a replay buffer to store previous experiences to later on sample random and uncorrelated mini batches to learn from.

2. _Target Networks_: these networks will be a time-delayed copies of the original ones that slowly track the learned networks. They prevent the "chasing a moving target" effect when using the same parameters (weights) for estimating the target and the Q value. This is because there is a big correlation between the TD target and the parameters we are changing. These target networks are [softly updated](https://github.com/jscriptcoder/Multi-Agent-Collaboration-and-Competition/blob/master/agent/utils.py#L17) by polyak averaging: <img src="images/polyak_avg.svg" />

The Q-Network is updated by minimizing the mean squared Bellman equation as followed:

<p align="center"><img src="images/loss_q.svg" /></p>

where ```(s,r,a,s',d)~D``` are random mini batches of transitions, and ```d``` indicates whether state ```s'``` is terminal.

For our policy function, our objective is to maximize the expected return, so we want to learn a ```μ(s)``` that maximizes ```Q(s,a)```. Now, remember that our action space is continuous, so we're assuming that the Q-function is differentiable with respect to action, we can then simply perform gradient ascent with respect to policy parameters to solve:


<p align="center"><img src="images/max_policy.svg" /></p>

TODO

Sources: [OpenAI, Spinning up, Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)

**Twin Delayed DDPG or TD3**:

TODO

Sources: [OpenAI, Spinning up, Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html), [TD3: Learning To Run With AI](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93)

**Soft Actor Critic or SAC**:

TODO

Sources: [OpenAI, Spinning up, Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html), [Soft Actor-Critic Demystified](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665)

### Neural Networks Architecture

TODO

## Plot of Rewards

1. **DDPG**:

<p align="center"><img src="images/ddpg_avg_score.png" /></p>

As seen in the [jupyter notebook](Tennis.ipynb), the environment was solved, 100 times consecutively, with an avarage score of 0.9, after 918 episodes and 0:15:53 running on CPU.

2. **TD3**:

<p align="center"><img src="images/td3_avg_score.png" /></p>

The environment was solved with an avarage score of 2.125, after 1716 episodes and 0:42:42 running on CPU.

2. **SAC**:

<p align="center"><img src="images/sac_avg_score.png" /></p>

The environment was solved with an avarage score of 1.727, after 1570 episodes and 0:41:21 running on CPU.

## Ideas for Future Work
TODO
