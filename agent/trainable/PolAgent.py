import os
import torch

import initialize
from agent.trainable.base import TrainableAgent


class PolAgent(TrainableAgent):
    """ Deterministic System Stochastic Action Agent"""

    def __init__(self, OperationModel):
        super(PolAgent, self).__init__()

        self.OperationModel = OperationModel

        self.operation_pol = None

        self.env_spaces = None
        self.env_param_space = None
        self.env_param = None
        self.horizon = None

    def reset_parameters(self):
        """ reset the models parameters """
        self.operation_pol.reset_parameters()

    def to(self, device):
        """Put the models on a device"""
        self.operation_pol.to(device)
        self.env_param = self.env_param.to(device)

    def get_batch_systems(self, nb_systems):
        """Make a decision concerning a batch of the system"""
        return torch.cat([self.get_system() for _ in range(nb_systems)])

    def get_system(self):
        """Make a decision concerning the system"""
        return self.env_param

    def set_system(self, parameters):
        """Changes the constant set of parameters"""
        self.env_param = parameters

    def get_action(self, observation):
        """Make an operational decision"""
        return self.operation_pol(observation).sample()

    def get_action_log_prob(self, observation, action):
        """Get loglikelihood of an operational decision"""
        return self.operation_pol(observation).log_prob(action)

    def get_entropy(self, observation):
        """Get entropy of the distribution"""
        return self.operation_pol(observation).entropy()

    def initialize(self, env, **kwargs):
        """ Initializes the agent from the environment """
        # save the spaces
        self.env_spaces = env.spaces
        self.env_param_space = env.parameter_space
        self.env_param = env.get_parameters()

        # get the horizon
        self.horizon = env.horizon

        # instantiate the policy model
        if isinstance(self.OperationModel, str):
            self.operation_pol = initialize.get_model(env, self.OperationModel, kwargs["operation_pol"])
        else:
            self.operation_pol = self.OperationModel(env, **kwargs["operation_pol"])

        return self

    def project_parameters(self):
        """Project the parameters on their feasible set"""
        pass

    def get_operation_parameters(self):
        """Returns the parameters of the operation policy"""
        return self._get_operation_model().parameters()

    def _get_operation_model(self):
        """Returns the the investment model"""
        return self.operation_pol

    def save(self, path):
        model, path_model = self._path_iterate(path)

        dir, file = os.path.split(path_model)
        if dir != '':
            os.makedirs(dir, exist_ok=True)  # required if directory not created yet

        torch.save(model.state_dict(), path_model)

    def load(self, path):
        model, path_model = self._path_iterate(path)
        model.load_state_dict(torch.load(path_model))

    def _path_iterate(self, path):
        return self.operation_pol, os.path.join(path, f'agent-save/operation-pol/operation')
