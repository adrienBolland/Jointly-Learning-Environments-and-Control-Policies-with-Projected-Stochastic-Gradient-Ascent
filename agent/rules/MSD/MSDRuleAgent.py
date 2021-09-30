from abc import ABC, abstractmethod
import os

import torch
from torch.distributions import Bernoulli

from agent.base import BaseAgent


class MSDBaseRule(BaseAgent, ABC):
    def __init__(self):
        super(MSDBaseRule, self).__init__()
        self.nb_actions = None
        self.env_param = None
        self.equilibrium = None

    def get_batch_systems(self, nb_systems):
        """Make a decision concerning a batch of the system"""
        return torch.cat([self.get_system() for _ in range(nb_systems)])

    def get_system(self):
        return self.env_param

    def set_system(self, parameters):
        self.env_param = parameters

    @abstractmethod
    def get_action(self, observation):
        raise NotImplementedError

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.n
        self.env_param = env.get_parameters()
        self.equilibrium = env.unwrapped.equilibrium
        return self

    def save(self, path):
        """Save models"""
        path = os.path.join(path, f'agent-save/investment-pol/investment')

        dir, file = os.path.split(path)
        if dir != '':
            os.makedirs(dir, exist_ok=True)  # required if directory not created yet

        torch.save(self.env_param, path)

    def load(self, path):
        """Load models"""
        path = os.path.join(path, f'agent-save/investment-pol/investment')
        self.env_param = torch.load(path)


class MSDRuleAgent(MSDBaseRule):

    def get_action(self, observation):
        positions, _ = observation.split(1, dim=-1)
        actions = torch.empty_like(positions, dtype=torch.long)

        # if x_t < x_eq, then play action number zero (-0.3)
        actions[positions > self.equilibrium] = 0

        # if x_t >= x_eq, then play action number 2 (0.0)
        actions[positions <= self.equilibrium] = self.nb_actions // 2

        return actions


class MSDRandomRuleAgent(MSDBaseRule):

    def __init__(self):
        super(MSDRandomRuleAgent, self).__init__()
        self.actions_value = None

    def get_action(self, observation):
        positions, _ = observation.split(1, dim=-1)
        actions = torch.empty_like(positions, dtype=torch.long)

        # compute the two actions
        a_eq = self.env_param[0, 1].pow(3) * self.equilibrium

        a_lower = self.actions_value[self.actions_value < a_eq]
        a_lower = a_lower.argmax() if len(a_lower) else torch.tensor(0)

        a_upper = self.actions_value[self.actions_value >= a_eq]
        a_upper = a_upper.argmin() if len(a_upper) else torch.tensor(self.nb_actions - 1)

        p_lower = torch.ones_like(a_lower).float()
        p_upper = torch.zeros_like(a_upper).float()

        p_lower[a_upper != a_lower] = (a_eq - a_lower) / (a_upper - a_lower)
        p_upper[a_upper != a_lower] = (a_upper - a_eq) / (a_upper - a_lower)

        # draw random actions
        actions_id = Bernoulli(probs=torch.tensor([p_lower])).sample(positions.shape[:-1])
        actions[actions_id == 0] = a_lower
        actions[actions_id == 1] = a_upper

        return actions

    def initialize(self, env, **kwargs):
        super(MSDRandomRuleAgent, self).initialize(env, **kwargs)
        self.actions_value = env.unwrapped.actions_value
        return self
