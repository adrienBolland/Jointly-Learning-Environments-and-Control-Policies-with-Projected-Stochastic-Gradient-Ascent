import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from model.decision.ForwardDistModel import ForwardDistModel


class GaussianDistModel(ForwardDistModel):

    def __init__(self, sys, layers, act_fun=nn.Tanh, correlated=False, scale=None):
        self.nb_actions = sys.action_space.shape[0]

        super(GaussianDistModel, self).__init__(
            input_size=sys.observation_space.shape[0],
            output_size=self.nb_actions * (self.nb_actions + 3) // 2 if correlated else 2 * self.nb_actions,
            layers=layers,
            act_fun=act_fun)

        self.correlated = correlated
        self.id_lower = torch.tril_indices(self.nb_actions, self.nb_actions, -1)

        self.scale = scale

    def forward(self, x):
        features = self.net(x)
        mean, std, cov = features.split((self.nb_actions, self.nb_actions, self.output_size - 2 * self.nb_actions),
                                        dim=-1)

        scale_tril = torch.diag_embed(std.abs())

        if self.correlated:
            scale_tril[..., self.id_lower[0, :], self.id_lower[1, :]] = cov

        covariance_matrix = torch.matmul(scale_tril, scale_tril.transpose(-2, -1))
        constant = 10e-6 * torch.diag_embed(torch.ones_like(std))
        return MultivariateNormal(mean, covariance_matrix=covariance_matrix+constant)

    def reset_parameters(self):
        super(GaussianDistModel, self).reset_parameters()

        # scale the last layer by a factor
        if self.scale is not None:
            with torch.no_grad():
                last_layer_w = self.layers[-1].weight
                last_layer_w.copy_(last_layer_w / self.scale)
