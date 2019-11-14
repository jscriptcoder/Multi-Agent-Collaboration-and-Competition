import numpy as np
from unityagents import UnityEnvironment
import torch.nn.functional as F
from torch.optim import Adam
from agent.ddpg_agent import DDPGAgent
from agent.td3_agent import TD3Agent
from agent.multi_agent import MultiAgent

env = UnityEnvironment(file_name='./env/Tennis.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

from agent.config import Config
from agent.utils import EnvironmentAdapterForUnity, play_game
unity_env = EnvironmentAdapterForUnity(env, brain_name)
unity_env.train_mode=False
config = Config()

# Common
config.seed = 42
config.num_agents = 2
config.env = unity_env
config.times_solved = 1
config.buffer_size = int(1e6)
config.state_size = state_size
config.action_size = action_size
config.use_ou_noise = False

# DDPG
#config.hidden_actor = (64, 64)
#config.hidden_critic = (512,)
#config.lr_actor = 1e-3
#config.lr_critic = 1e-3
#config.activ_actor = F.relu
#config.activ_critic = F.relu
#config.optim_actor = Adam
#config.optim_critic = Adam
#
#ml_agent = MultiAgent(DDPGAgent, config)
#ml_agent.load_agents_weights()
#play_game(ml_agent, config.env_solved)

# TD3
#config.hidden_actor = (64, 64)
#config.hidden_critic = (256, 256)
#config.lr_actor = 1e-3
#config.lr_critic = 1e-3
#config.activ_actor = F.relu
#config.activ_critic = F.relu
#config.optim_actor = Adam
#config.optim_critic = Adam
#
#ml_agent = MultiAgent(TD3Agent, config)
#ml_agent.load_agents_weights()
#play_game(ml_agent, config.env_solved)

# SAC
from agent.multi_sac_agent import MultiSacAgent

config.hidden_actor = (64, 64)
config.hidden_critic = (512,)
config.lr_actor = 1e-3
config.lr_critic = 1e-3
config.activ_actor = F.relu
config.activ_critic = F.relu
config.optim_actor = Adam
config.optim_critic = Adam
config.log_std_min=-20
config.log_std_max=2
config.alpha_auto_tuning = False

ml_agent = MultiSacAgent(config)
ml_agent.load_agents_weights()

play_game(ml_agent, config.env_solved)