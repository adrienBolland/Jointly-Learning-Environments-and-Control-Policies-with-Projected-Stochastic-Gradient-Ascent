from agent.base import BaseAgent

import torch


class DiscreteRandomAgent(BaseAgent):

    def __init__(self):
        super(DiscreteRandomAgent, self).__init__()
        self.nb_actions = None
        self.env_param = None

    def get_system(self):
        return self.env_param

    def get_action(self, observation):
        return torch.randint(0, self.nb_actions, (observation.shape[0], 1))

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.n
        self.env_param = env.get_parameters()
        return self

    def save(self, path):
        """Save models"""
        pass

    def load(self, path):
        """Load models"""
        pass


class NormalRandomAgent(BaseAgent):

    def __init__(self):
        super(NormalRandomAgent, self).__init__()
        self.nb_actions = None
        self.env_param = None

    def get_system(self):
        return self.env_param

    def get_action(self, observation):
        return torch.randn((observation.shape[0], self.nb_actions))

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.shape[0]
        self.env_param = env.get_parameters()
        return self

    def save(self, path):
        """Save models"""
        pass

    def load(self, path):
        """Load models"""
        pass
