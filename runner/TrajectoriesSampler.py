import torch

from runner.base import BaseRunner


class TrajectoriesSampler(BaseRunner):

    def __init__(self, sys, agent):
        super(TrajectoriesSampler, self).__init__(sys=sys, agent=agent)

    def sample(self, nb_trajectories):
        # get system
        param = self.set_system()
        param_batches = torch.repeat_interleave(param, nb_trajectories, dim=0)

        return *self._sample(nb_trajectories), param_batches
