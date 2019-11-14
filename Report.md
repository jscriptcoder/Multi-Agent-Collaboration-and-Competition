# Report: Tennis environment using Multi-Agent approach

## Learning Algorithm

What we're dealing with here is an envirornment with continuous observation space that consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm, and continues action space, a vector with 4 numbers, corresponding to torque applicable to two joints. Policy Gradient methods are the right fit for continuos action space.

I'll try to solve this environment with three state-of-the-art DRL algorithms, using the latest Actor-Critic methods:

- Deep Deterministic Policy Gradient (a.k.a DDPG). [Paper](https://arxiv.org/abs/1509.02971)
- Twin Delayed DDPG or TD3. [Paper](https://arxiv.org/abs/1802.09477)
- Soft Actor Critic or SAC. [First paper](https://arxiv.org/abs/1801.01290). [Second paper](https://arxiv.org/abs/1812.05905) improving on the first one

### Hyperparameters

Following is a list of all the hyperparameters used and their values:

#### General params
- ```seed = 0```

TODO

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
