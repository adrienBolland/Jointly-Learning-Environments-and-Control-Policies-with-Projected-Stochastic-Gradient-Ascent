import torch.nn as nn
from torch.distributions import OneHotCategorical

from model.decision.ForwardDistModel import ForwardDistModel


class CategoricalDistModel(ForwardDistModel):

    def __init__(self, sys, layers, act_fun=nn.Tanh):
        super(CategoricalDistModel, self).__init__(input_size=sys.observation_space.shape[0],
                                                   output_size=sys.action_space.n,
                                                   layers=layers,
                                                   act_fun=act_fun)

    def forward(self, x):
        return OneHotCategorical(logits=self.net(x))
