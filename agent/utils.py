import gym
import time
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from .device import device

gym.logger.set_level(40)

def hidden_init(layer):
    """Will return a tuple with the range for the initialization
    of hidden layers

    Args:
        layer (torch.nn.Layer)

    Returns:
        Tuple of int
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Args:
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Helper to create a experience tuple with named fields
make_experience = namedtuple('Experience',
                             field_names=['state',
                                          'action',
                                          'reward',
                                          'next_state',
                                          'done'])

def from_experience(experiences):
    """Returns a tuple with (s, a, r, s', d)

    Args:
        namedtuple: List of tensors

    Returns:
        Tuple of torch.Tensor
    """
    # tensor(batch_size, num_agents, state_size)
    states = torch.from_numpy(
            np.array([e.state for e in experiences if e is not None]))\
            .float().to(device)

    # tensor(batch_size, num_agents, action_size)
    actions = torch.from_numpy(
            np.array([e.action for e in experiences if e is not None]))\
            .float().to(device)

    # tensor(batch_size, num_agents)
    rewards = torch.from_numpy(
            np.array([e.reward for e in experiences if e is not None]))\
            .float().to(device)

    # tensor(batch_size, num_agents, state_size)
    next_states = torch.from_numpy(
            np.array([e.next_state for e in experiences if e is not None]))\
            .float().to(device)

    # tensor(batch_size, num_agents)
    dones = torch.from_numpy(
            np.array([e.done for e in experiences if e is not None])\
            .astype(np.uint8)).float().to(device)

    return states, actions, rewards, next_states, dones

def run_env(env, get_action=None, max_t=1000, close_env=True):
    """Run actions against an environment.
    We pass a function in that could or not be wrapping an agent's actions

    Args:
        env (Environment)
        get_action (func): returns actions based on a state
        max_t (int): maximum number of timesteps
    """

    if get_action is None:
        get_action = lambda _: env.action_space.sample()

    state = env.reset()
    env.render()

    while True:
        action = get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()

        if done: break

    if close_env:
        env.close()


class EnvironmentAdapterForUnity():
    """Wrapper for Unity Environment.
    The idea is to have common interface for all the environments:
        OpenAI Gym envs, Unity envs, etc...

    Args:
        unity_env (UnityEnvironment)
        brain_name (str):
            name of the brain responsible for deciding the actions of their
            associated agents

    Attributes:
        unity_env (UnityEnvironment)
        brain_name (str)
        train_mode (bool)
    """

    def __init__(self, unity_env, brain_name):
        self.unity_env = unity_env
        self.brain_name = brain_name
        self.train_mode = True

        brain = unity_env.brains[brain_name]

        self.action_size = brain.vector_action_space_size
        self.state_size = brain.vector_observation_space_size

    def render(self):
        pass # Unity envs don't need to render

    def step(self, action):
        env_info = self.unity_env.step(action)[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = done = env_info.local_done

        return next_state, reward, done, env_info

    def reset(self):
        env_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        return env_info.vector_observations

    def close(self):
        self.unity_env.close()


def scores2poly1d(scores, polyfit_deg):
    """Fit a polynomial to a list of scores

    Args:
        scores (List of float)
        polyfit_deg (int): degree of the fitting polynomial

    Returns:
        List of int, one-dimensional polynomial class
    """
    x = list(range(len(scores)))
    degs = np.polyfit(x, scores, polyfit_deg)
    return x, np.poly1d(degs)


def plot_scores(scores,
                title='Agents avg score',
                figsize=(15, 6),
                polyfit_deg=None):
    """Plot scores over time. Optionally will draw a line showing the trend

    Args:
        scores (List of float)
        title (str)
        figsize (tuple of float)
            Default: (15, 6)
        polyfit_deg (int): degree of the fitting polynomial (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(scores)

    max_score = max(np.round(scores, 3))
    idx_max = np.argmax(scores)
    plt.scatter(idx_max, max_score, c='r', linewidth=3)

    if polyfit_deg is not None:
        x, p = scores2poly1d(scores, polyfit_deg)
        plt.plot(p(x), linewidth=3)

    plt.title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Epochs')
    ax.legend(['Score', 'Trend', 'Max avg score: {}'.format(max_score)])

def get_time_elapsed(start, end=None):
    """Returns a human readable (HH:mm:ss) time difference between two times

    Args:
        start (float)
        end (float): optional value
            Default: now
    """

    if end is None:
        end = time.time()
    elapsed = round(end-start)
    return str(datetime.timedelta(seconds=elapsed))

def play_game(ml_agent, min_score):
    """Will play a game until it reaches the min_score

    Args:
        ml_agent (MultiAgent)
        min_score (float)
    """
    score = -np.inf
    while score < min_score:
        score = ml_agent.eval_episode()
