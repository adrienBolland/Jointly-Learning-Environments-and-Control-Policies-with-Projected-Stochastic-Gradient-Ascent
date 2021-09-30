from abc import ABC, abstractmethod

import torch

from algo.joint import utils
from algo.joint.base import BaseJointPGOnline
from runner.TrajectoriesSampler import TrajectoriesSampler


class DESGA(BaseJointPGOnline, ABC):
    """ Direct Environment Search with Gradient Ascent """

    def __init__(self, env, agent):
        super(DESGA, self).__init__(env=env, agent=agent)

    @abstractmethod
    def loss(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch):
        """ Loss function for computing the gradient """
        raise NotImplementedError

    def _get_runner(self):
        """ get the specific runner for sampling trajectories"""
        return TrajectoriesSampler(self.env, self.agent)


class ReinforceDESGA(DESGA):

    def __init__(self, env, agent):
        super(ReinforceDESGA, self).__init__(env=env, agent=agent)

        self.partial_clip = None

    def initialize(self, **kwargs):
        super(ReinforceDESGA, self).initialize(**kwargs)

        # do we clip the gradients
        self.partial_clip = kwargs.get("partial_clip", False)

    def loss(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch):
        """ extended reinforce loss """
        with torch.no_grad():
            mean_rew = torch.mean(torch.sum(reward_batch, dim=1)).squeeze().item()

        # set the system for having the gradients w.r.t the env parameters
        self.env.set_parameters(self.agent.get_system())

        states_batch_grad = torch.zeros(state_batch.shape, device=self.env.unwrapped.device)
        states_batch_grad[:, 0, :] = state_batch[:, 0, :]

        for t in range(self.env.horizon - 1):
            s_t_grad = self.env.dynamics(states_batch_grad[:, t, :].clone(),
                                         action_batch[:, t, :],
                                         dist_batch[:, t, :])
            if self.partial_clip:
                s_t_grad = utils.ClipGrad.apply(s_t_grad)

            states_batch_grad[:, t + 1, :] = s_t_grad

        # compute the error
        batch_size = state_batch.shape[0]
        loss_batch = []

        for batch in range(batch_size):
            err = self._system_loss(states_batch_grad[batch, :-1, :],
                                    dist_batch[batch, :, :],
                                    action_batch[batch, :, :],
                                    reward_batch[batch, :, :],
                                    mean_rew)
            loss_batch.append(err)

        loss = torch.mean(torch.stack(loss_batch))

        return loss, loss

    def _system_loss(self, states, disturbances, actions, rew, baseline=0.):
        # loglikelihood of the actions
        sum_log_pol_grad = torch.sum(self.agent.get_action_log_prob(states, actions))

        # loglikelihood of the disturbances
        sum_log_dist_grad = torch.sum(self.env.disturbance(states, actions).log_prob(disturbances))

        # sum of the loglikelihood
        sum_log_p_grad = sum_log_pol_grad + sum_log_dist_grad

        # cumulative reward
        rew_grad = self.env.reward(states, actions, disturbances)

        return -(sum_log_p_grad * (torch.sum(rew) - baseline) + torch.sum(rew_grad))
