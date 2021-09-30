from agent.base import BaseAgent

import torch


class ConstantActionAgent(BaseAgent):

    def __init__(self, action):
        super(ConstantActionAgent, self).__init__()
        self.nb_actions = None
        self.env_param = None
        self.action = action

    def get_system(self):
        return self.env_param

    def get_action(self, observation):
        return torch.empty((observation.shape[0], self.nb_actions)).fill_(self.action)

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
