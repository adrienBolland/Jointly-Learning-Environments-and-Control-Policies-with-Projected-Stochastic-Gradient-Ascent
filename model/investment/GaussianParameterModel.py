import torch
from torch import nn
from torch.distributions import Normal


class GaussianParameterModel(nn.Module):

    def __init__(self, sys, feasible_set, init=None):
        super(GaussianParameterModel, self).__init__()

        self.gaussian_mean = nn.Parameter(torch.Tensor(1, sys.parameter_space.shape[0]), requires_grad=True)
        self.gaussian_std = nn.Parameter(torch.Tensor(1, sys.parameter_space.shape[0]), requires_grad=True)
        self.feasible_set = feasible_set
        self.init = init

        self.reset_parameters()

    def reset_parameters(self):
        if self.init is not None:
            for i, v in enumerate(self.init):
                nn.init.constant_(self.gaussian_mean[0, i], v)
                nn.init.constant_(self.gaussian_std[0, i], 1e-04)
        else:
            for i, bounds in enumerate(self.feasible_set):
                nn.init.uniform_(self.gaussian_mean[0, i],
                                 self.feasible_set[bounds][0],
                                 self.feasible_set[bounds][1])
                nn.init.constant_(self.gaussian_std[0, i],
                                  (self.feasible_set[bounds][1] - self.feasible_set[bounds][0]) / 3.)

    def project_parameters(self):
        pass

    def forward(self):
        """ returns a vector of parameters """
        return Normal(loc=self.gaussian_mean, scale=self.gaussian_std ** 2 + 1e-04)
