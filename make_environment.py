
import gymnasium as gym
import random
from gymnasium.wrappers import RescaleAction
from typing import Optional
import numpy as np
import torch
from dmcontrol_environment import *
from get_model import *

class SinglePrecision(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
        
class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)

def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             terminate_when_unhealthy: bool = True,
             action_concat: int = 1,
             obs_concat: int = 1,
             continuous: bool = True,
             ) -> gym.Env:

    # Check if the env is in gym.
    env_ids = list(gym.envs.registry.keys())

    domain_name, task_name = env_name.split('-')
    env = DMCEnv(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
        env = FlattenAction(env)

    if continuous:
        env = RescaleAction(env, -1.0, 1.0)

    env = SinglePrecision(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env

class Experiment(object):
    def __init__(self):
        #self.args = args
        self.n_total_steps = 0
        self.max_steps = 100000
        self.env = make_env('cartpole-swingup', 1)
        self.eval_env = make_env('cartpole-swingup', 101)
        self.agent = get_model( self.env)