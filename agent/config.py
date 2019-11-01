import torch.nn.functional as F
from torch.optim import Adam

class Config:
    seed = 0
    num_agents = 2
    env = None
    env_solved = 0.5
    times_solved = 100
    buffer_size = int(1e6)
    batch_size = 128
    num_episodes = 2000
    num_updates = 2
    max_steps = 2000
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
    grad_clip_actor = None
    grad_clip_critic = None
    use_huber_loss = False
    update_every = 4
    use_ou_noise = True
    ou_mu = 0.0
    ou_theta = 0.15
    ou_sigma = 0.2
    expl_noise = 0.1
    noise_weight = 1.0
    noise_decay = 0.99
    use_linear_decay = False
    noise_linear_decay = 1e-6
    log_every = 100
    end_on_solved = False
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq_update = 2
    
    