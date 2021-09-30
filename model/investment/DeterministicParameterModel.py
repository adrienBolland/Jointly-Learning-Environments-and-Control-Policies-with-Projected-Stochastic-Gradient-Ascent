import torch
from torch import nn


class DeterministicParameterModel(nn.Module):

    def __init__(self, sys, feasible_set, init=None):
        super(DeterministicParameterModel, self).__init__()

        self.parameter = nn.Parameter(torch.Tensor(1, sys.parameter_space.shape[0]), requires_grad=True)
        self.feasible_set = feasible_set
        self.init = init

        self.reset_parameters()

    def reset_parameters(self):
        if self.init is not None:
            for i, v in enumerate(self.init):
                nn.init.constant_(self.parameter[0, i], v)
        else:
            for i, bounds in enumerate(self.feasible_set):
                nn.init.uniform_(self.parameter[0, i], self.feasible_set[bounds][0], self.feasible_set[bounds][1])

    def project_parameters(self):
        for i, bounds in enumerate(self.feasible_set):
            # projected value
            with torch.no_grad():
                val = self.parameter[0, i].clamp(self.feasible_set[bounds][0], self.feasible_set[bounds][1]).item()

            # initialize the new value
            nn.init.constant_(self.parameter[0, i], val)

    def forward(self):
        """ returns a vector of parameters """
        return self.parameter
