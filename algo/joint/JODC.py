import torch

from algo.joint import utils
from algo.joint.base import BaseJointPG, BaseJointPGOnline
from runner.ParallelSystemsTrajectoriesSampler import ParallelSystemsTrajectoriesSampler


class ReinforceJODC(BaseJointPGOnline):

    def __init__(self, env, agent):
        super(ReinforceJODC, self).__init__(env=env, agent=agent)

    def loss(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch):
        """ extended reinforce loss """
        # compute the cumulative reward
        cum_r = torch.sum(reward_batch, dim=1)
        mean_cum_r = torch.mean(cum_r)

        # system loss
        if self.system_fit:
            log_p_param = self.agent.get_investment_log_prob(parameters_batch)
            system_loss = -torch.mean(torch.sum(log_p_param * (cum_r - mean_cum_r), dim=-1), dim=0)
        else:
            system_loss = None

        # policy loss
        if self.policy_fit:
            log_prob_a = self.agent.get_action_log_prob(state_batch[:, :-1, :], action_batch)
            policy_loss = -torch.mean(torch.sum(log_prob_a * (cum_r - mean_cum_r), dim=-1), dim=0)
        else:
            policy_loss = None

        return system_loss, policy_loss

    def _get_runner(self):
        """ get the specific runner for sampling trajectories"""
        return ParallelSystemsTrajectoriesSampler(self.env, self.agent)


class PPOJODC(BaseJointPG):

    def __init__(self, env, agent):
        super(PPOJODC, self).__init__(env=env, agent=agent)

        self.nb_epochs = None
        self.eps_clip = None

    def initialize(self, **kwargs):
        super(PPOJODC, self).initialize(**kwargs)

        # number of epochs
        self.nb_epochs = kwargs.get("nb_epochs", 1)

        # epsilon
        self.eps_clip = kwargs.get("eps_clip", 0.1)

    def _optimize(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch,
                  policy_optimizer, investment_optimizer):
        # dict of gradients
        grad_norm = dict()

        # first optimize the policy with PPO
        if self.policy_fit:
            # list of losses and gradeents
            policy_loss = []
            grad_norm["policy"] = 0.

            # old log likelihood
            old_log_p = self.agent.get_action_log_prob(state_batch[:, :-1, :], action_batch).detach()

            # compute the advantage
            scaled_reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
            advantages = scaled_reward_batch.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]).squeeze()

            # perform the updates
            for _ in range(self.nb_epochs):
                # zero grad
                policy_optimizer.zero_grad()

                # the probability ratio
                log_prob_a = self.agent.get_action_log_prob(state_batch[:, :-1, :], action_batch)
                ratios = torch.exp(log_prob_a - old_log_p)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                loss = -torch.mean(torch.sum(torch.min(surr1, surr2), dim=-1))
                policy_loss.append(loss.detach())

                loss.backward()
                grad_norm["policy"] += self.nb_epochs * utils.gradient_norm(self.agent.get_operation_parameters())
                policy_optimizer.step()
                self.agent.project_parameters()

            policy_loss = torch.stack(policy_loss).mean()
        else:
            policy_loss = None

        # system loss
        if self.system_fit:
            investment_optimizer.zero_grad()

            cum_r = torch.sum(reward_batch, dim=1)
            mean_cum_r = torch.mean(torch.sum(reward_batch, dim=1))
            log_p_param = self.agent.get_investment_log_prob(parameters_batch)
            system_loss = -torch.mean(torch.sum(log_p_param * (cum_r - mean_cum_r), dim=-1), dim=0)

            system_loss.backward()
            grad_norm["system"] = utils.gradient_norm(self.agent.get_investment_parameters())
            investment_optimizer.step()
            self.agent.project_parameters()
        else:
            system_loss = None

        # project the parameters on their feasible set
        self.agent.project_parameters()

        return system_loss, policy_loss, grad_norm

    def _get_runner(self):
        """ get the specific runner for sampling trajectories"""
        return ParallelSystemsTrajectoriesSampler(self.env, self.agent)
