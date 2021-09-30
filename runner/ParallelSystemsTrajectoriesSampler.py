import torch

from runner.base import BaseRunner


class ParallelSystemsTrajectoriesSampler(BaseRunner):

    def __init__(self, sys, agent):
        super(ParallelSystemsTrajectoriesSampler, self).__init__(sys=sys, agent=agent)

    def sample(self, nb_trajectories):
        # get system
        param_batches = self.set_batch_systems(nb_trajectories)

        return *self._sample(nb_trajectories), param_batches
