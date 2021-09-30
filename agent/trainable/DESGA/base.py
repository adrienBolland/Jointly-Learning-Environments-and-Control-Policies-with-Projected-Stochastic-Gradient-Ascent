import os
import torch

from abc import ABC, abstractmethod

import initialize
from agent.trainable.base import TrainableAgent


class DESGABaseAgent(TrainableAgent, ABC):
    """ agents making decisions about the system and about its control """

    def __init__(self, InvestmentModel, OperationModel):
        super(DESGABaseAgent, self).__init__()
        self.InvestmentModel = InvestmentModel
        self.OperationModel = OperationModel

        self.investment_pol = None
        self.operation_pol = None

        self.env_spaces = None
        self.env_param_space = None
        self.horizon = None

    def get_batch_systems(self, nb_systems):
        """Make a decision concerning a batch of the system"""
        return torch.cat([self.get_system() for _ in range(nb_systems)])

    def reset_parameters(self):
        """ reset the models parameters """
        self.investment_pol.reset_parameters()
        self.operation_pol.reset_parameters()

    def to(self, device):
        """Put the models on a device"""
        self.investment_pol.to(device)
        self.operation_pol.to(device)

    def initialize(self, env, **kwargs):
        """ Initializes the agent from the environment """
        # save the spaces
        self.env_spaces = env.spaces
        self.env_param_space = env.parameter_space

        # get the horizon
        self.horizon = env.horizon

        # instantiate the models
        if isinstance(self.InvestmentModel, str):
            self.investment_pol = initialize.get_model(env,
                                                       self.InvestmentModel,
                                                       {"feasible_set": env.get_feasible_set(),
                                                        **kwargs["investment_pol"]})
        else:
            self.investment_pol = self.InvestmentModel(env,
                                                       **{"feasible_set": env.get_feasible_set(),
                                                          **kwargs["investment_pol"]})

        if isinstance(self.OperationModel, str):
            self.operation_pol = initialize.get_model(env, self.OperationModel, kwargs["operation_pol"])
        else:
            self.operation_pol = self.OperationModel(env, **kwargs["operation_pol"])

        return self

    @abstractmethod
    def project_parameters(self):
        """Project the parameters on their feasible set"""
        raise NotImplementedError

    def get_investment_parameters(self):
        """Returns the parameters of the investment policy"""
        return self._get_investment_model().parameters()

    def get_operation_parameters(self):
        """Returns the parameters of the operation policy"""
        return self._get_operation_model().parameters()

    def _get_investment_model(self):
        """Returns the the investment model"""
        return self.investment_pol

    def _get_operation_model(self):
        """Returns the the investment model"""
        return self.operation_pol

    def save(self, path):
        for model, path_model in self._path_iterate(path):

            dir, file = os.path.split(path_model)
            if dir != '':
                os.makedirs(dir, exist_ok=True)  # required if directory not created yet

            torch.save(model.state_dict(), path_model)

    def load(self, path):
        for model, path_model in self._path_iterate(path):
            model.load_state_dict(torch.load(path_model))

    def _path_iterate(self, path):
        for model, name in zip([self.investment_pol, self.operation_pol], ['investment', 'operation']):
            if model is None:
                continue

            path_model = os.path.join(path, f'agent-save/{name}-pol/{name}')

            yield model, path_model
