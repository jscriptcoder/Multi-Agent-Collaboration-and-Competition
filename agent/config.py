import torch.nn.functional as F
from torch.optim import RMSprop, Adam

class Config:
    seed = 101
    num_agents = 2
    env = None
    env_solved = 0.5
    buffer_size = int(1e6)
    batch_size = 128
    num_episodes = 2000
    state_size = None
    action_size = None
    gamma = 0.99
    tau = 1e-3
    lr_actor = 1e-4
    lr_critic = 3e-4
    hidden_actor = (64, 64)
    hidden_critic = (64, 64)
    activ_actor = F.relu
    activ_critic = F.relu
    optim_actor = Adam
    optim_critic = Adam
    weight_decay = 0.0001 # L2 decay
    update_every = 4
    ou_mu = 0.0
    ou_theta = 0.15
    ou_sigma = 0.2
    noise_weight = 1.0
    noise_decay = 0.99
    