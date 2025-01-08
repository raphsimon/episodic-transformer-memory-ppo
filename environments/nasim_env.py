import gymnasium as gym
import numpy as np
import time

from nasim.envs.wrappers import StochasticEpisodeStarts
from gymnasium.wrappers import StepAPICompatibility

class NASimWrapper:
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self._env = StochasticEpisodeStarts(self._env)
        self._env = StepAPICompatibility(self._env, output_truncation_bool=False)
        self.max_episode_steps = self._env.unwrapped.scenario.step_limit
        # Whether to make CartPole partial observable by masking out the velocity.

    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs, reward, done, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()