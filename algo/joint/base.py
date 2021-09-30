from abc import ABC, abstractmethod

import re

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from algo import VERBOSE
from algo.base import BaseAlgo
from algo.joint import utils


class BaseJointPG(BaseAlgo, ABC):
    """ Joint Optimization of Design and Control (https://ttic.uchicago.edu/~cbschaff/nlimb/) """

    def __init__(self, env, agent):
        super(BaseJointPG, self).__init__(env=env, agent=agent)

        # parameters to be initialized
        self.optimizer_parameters = None
        self.optimizer_policy_parameters = None
        self.optimizer_investment_parameters = None
        self.scheduler_policy_lr = dict()
        self.scheduler_investment_lr = dict()
        self.batch_size = None
        self.nb_iterations = None
        self.mc_samples = None
        self.policy_fit = None
        self.system_fit = None
        self.policy_clip_norm = None
        self.system_clip_norm = None

    def initialize(self, **kwargs):
        # adam optimizer parameters
        self.optimizer_parameters = kwargs.get("optimizer_parameters", {"lr": 0.001})

        # adam optimizer parameters if two different hyper parameters
        self.optimizer_policy_parameters = kwargs.get("optimizer_policy_parameters", None)
        self.optimizer_investment_parameters = kwargs.get("optimizer_investment_parameters", None)

        # learning rate scheduler
        self.scheduler_policy_lr = kwargs.get("scheduler_policy_lr", {"milestones": []})
        self.scheduler_investment_lr = kwargs.get("scheduler_investment_lr", {"milestones": []})

        # batch size
        self.batch_size = kwargs.get("batch_size", 1)

        # number of iterations during the learning process
        self.nb_iterations = kwargs.get("nb_iterations")

        # number of Monte-Carlo samples for estimating the return
        self.mc_samples = kwargs.get("mc_samples", 32)

        # what parameters shall be fitted
        self.policy_fit = kwargs.get("policy_fit", True)
        self.system_fit = kwargs.get("system_fit", True)

        # clip norm of the gradient
        self.policy_clip_norm = kwargs.get("policy_clip_norm", None)
        self.system_clip_norm = kwargs.get("system_clip_norm", None)

    def fit(self, log_writer=None, device=None):
        # check the device
        if device is not None and device != "cpu":
            if not re.match(r"cuda:[0-9]+$", device):
                print(f"Warning: no such device as {device}, cpu will be used")
                device = "cpu"
            elif not torch.cuda.is_available():
                print(f"Warning: Cuda not available, cpu will be used")
                device = "cpu"

        # put the system and the agent on the device
        self.env.to(device)
        self.agent.to(device)

        # initialize the two optimizers and schedulers
        policy_optimizer = None
        investment_optimizer = None

        scheduler_policy_lr = None
        scheduler_investment_lr = None

        if self.policy_fit:
            param_kwarg = self.optimizer_policy_parameters if self.optimizer_policy_parameters is not None\
                else self.optimizer_parameters
            policy_optimizer = Adam(self.agent.get_operation_parameters(), **param_kwarg)

            scheduler_policy_lr = MultiStepLR(policy_optimizer, **self.scheduler_policy_lr)
        if self.system_fit:
            param_kwarg = self.optimizer_investment_parameters if self.optimizer_investment_parameters is not None\
                else self.optimizer_parameters
            investment_optimizer = Adam(self.agent.get_investment_parameters(), **param_kwarg)

            scheduler_investment_lr = MultiStepLR(policy_optimizer, **self.scheduler_investment_lr)

        if not self.system_fit and not self.policy_fit:
            # no parameters to learn
            return

        # create a runner for sampling trajectories
        runner = self._get_runner()

        for it in range(self.nb_iterations):
            loss = {}
            params = {}

            # set gradients to zero
            if self.policy_fit:
                policy_optimizer.zero_grad()
            if self.system_fit:
                investment_optimizer.zero_grad()

            # generate the batch
            with torch.no_grad():
                state_batch, dist_batch, reward_batch, action_batch, parameters_batch = runner.sample(self.batch_size)
                avg_rew = runner.cumulative_reward(reward_batch)

            # optimize the parameters
            system_loss, policy_loss, grad_norm = self._optimize(state_batch, dist_batch, reward_batch, action_batch,
                                                                 parameters_batch, policy_optimizer,
                                                                 investment_optimizer)

            # schedule the lr
            if self.policy_fit:
                scheduler_policy_lr.step()
            if self.system_fit:
                scheduler_investment_lr.step()

            # if verbose, do the print in the terminal
            if VERBOSE:
                print(f"Iteration {it} : performance = {avg_rew}")

            # plot info in the log
            with torch.no_grad():
                if self.system_fit and log_writer is not None:
                    params["system"] = self.agent._get_investment_model().named_parameters()

                    parameters_dict = self.env.parameters_dict()
                    parameters_mean = torch.mean(parameters_batch, dim=0).tolist()
                    parameters_mean_dict = {name: value for name, value in zip(parameters_dict, parameters_mean)}
                    log_writer.add_system_parameters(parameters_mean_dict, step=it)

                    loss["loss-system"] = system_loss.item()

                if self.policy_fit and log_writer is not None:
                    params["policy"] = self.agent._get_operation_model().named_parameters()
                    log_writer.add_policy_histograms(action_batch.view(-1, action_batch.shape[2]), step=it)

                    loss["loss-policy"] = policy_loss.item()

                if log_writer is not None:

                    log_writer.add_grad_histograms(params, step=it)
                    log_writer.add_loss(loss, step=it)
                    log_writer.add_grad(grad_norm, step=it)

                    # performance of the agent on the epoch
                    log_writer.add_expected_return(avg_rew, step=it)

                    # performance of the controller only
                    control_perf = self.env.control_perf(state_batch, action_batch, dist_batch, reward_batch)
                    log_writer.add_control_performance(control_perf, step=it)

    @abstractmethod
    def _optimize(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch,
                  policy_optimizer, investment_optimizer):
        """ Update the parameters """
        raise NotImplementedError

    @abstractmethod
    def _get_runner(self):
        """ get the specific runner for sampling trajectories"""
        raise NotImplementedError


class BaseJointPGOnline(BaseJointPG, ABC):

    def __init__(self, env, agent):
        super(BaseJointPGOnline, self).__init__(env=env, agent=agent)

        self.entropy_penalize = None

    def initialize(self, **kwargs):
        super(BaseJointPGOnline, self).initialize(**kwargs)

        self.entropy_penalize = EntropyPenalization(self.env, self.agent)
        self.entropy_penalize.initialize(**kwargs.get("entropy_init", dict()))

    def _optimize(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch,
                  policy_optimizer, investment_optimizer):
        # compute the losses
        system_loss, policy_loss = self.loss(state_batch, dist_batch, reward_batch, action_batch, parameters_batch)
        system_loss, policy_loss = self.entropy_penalize.penalize(system_loss, policy_loss, state_batch, dist_batch,
                                                                  reward_batch, action_batch, parameters_batch)

        self.entropy_penalize.step()

        # compute the gradients
        grad_norm = dict()

        if self.policy_fit:
            policy_loss.backward(retain_graph=self.system_fit)
            if self.policy_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.get_operation_parameters(), max_norm=self.policy_clip_norm)

            grad_norm["policy"] = utils.gradient_norm(self.agent.get_operation_parameters())
        if self.system_fit:
            system_loss.backward()
            if self.system_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.get_investment_parameters(), max_norm=self.system_clip_norm)

            grad_norm["system"] = utils.gradient_norm(self.agent.get_investment_parameters())

        # perform the gradient ascent
        if self.policy_fit:
            policy_optimizer.step()
        if self.system_fit:
            investment_optimizer.step()

        # project the parameters on their feasible set
        self.agent.project_parameters()

        return system_loss, policy_loss, grad_norm

    @abstractmethod
    def loss(self, state_batch, dist_batch, reward_batch, action_batch, parameters_batch):
        """ Loss function for computing the gradient """
        raise NotImplementedError


class EntropyPenalization:

    def __init__(self, sys, agent):
        self.sys = sys
        self.agent = agent

        self.weight = 0.
        self.decay = 1.

    def initialize(self, **kwargs):
        self.weight = kwargs.get("weight", 0.)
        self.decay = kwargs.get("decay", 1.)

    def penalize(self, system_loss, policy_loss, state_batch, dist_batch, reward_batch, action_batch, parameters_batch):
        entropy = torch.sum(self.agent.get_entropy(state_batch)) if self.weight > 0. else 0.
        return system_loss, policy_loss + self.weight * entropy

    def step(self):
        self.weight *= self.decay
