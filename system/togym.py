import torch

from gym import Env


class GymEnv(Env):

    def __init__(self, base_env):
        self._base_env = base_env

        self._state = None
        self._time = None

    def step(self, action):
        self._state, _, reward, _ = self._base_env(self._state, torch.tensor([action]))
        self._time += 1

        return self.state, reward.squeeze().item(), self._time >= self._base_env.horizon, dict()

    def reset(self):
        self._time = 0
        self._state = self._base_env.initial_state(1)
        return self.state

    def render(self, mode='human'):
        pass

    @property
    def state(self):
        return self._state.squeeze(dim=0).numpy()

    @property
    def action_space(self):
        return self._base_env.action_space

    @property
    def observation_space(self):
        return self._base_env.observation_space

    @property
    def reward_range(self):
        return -float("INF"), float("INF")
