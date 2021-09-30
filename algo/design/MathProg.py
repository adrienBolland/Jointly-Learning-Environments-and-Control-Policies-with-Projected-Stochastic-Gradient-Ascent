from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import numpy as np
import scipy.optimize as scp

import initialize
from algo import VERBOSE, VERY_VERBOSE
from algo.base import BaseAlgo
from runner.TrajectoriesSampler import TrajectoriesSampler


class MathProgBase(BaseAlgo, ABC):

    def __init__(self, env, agent):
        super(MathProgBase, self).__init__(env=env, agent=agent)

        self.optim_param = None
        self.policy_algo = None
        self.init_algo = None
        self.mc_samples = None
        self.res = None

        self.it = 0

    def initialize(self, **kwargs):
        # method for the parameters minimization
        self.optim_param = kwargs.get("optim_param", dict())

        # algo for optimizing the policy
        self.policy_algo = kwargs.get("policy_algo", None)
        self.init_algo = kwargs.get("init_algo", dict())

        # number of Monte-Carlo samples for estimating the return
        self.mc_samples = kwargs.get("mc_samples", 32)

    @abstractmethod
    def fit(self, log_writer=None, device=None):
        raise NotImplementedError

    def get_res(self):
        return self.res

    def _get_policy(self, parameters, log_writer):
        # get a copy of the agent
        agent = deepcopy(self.agent)
        agent.set_system(torch.tensor([parameters]).float())

        # set the system
        self.env.set_parameters(torch.tensor([parameters]).float())

        # fit an agent if required
        self._fit_if_required(self.env, agent)

        # estimate the agent's return
        runner = TrajectoriesSampler(self.env, agent)
        runner.set_system()

        state_batch, dist_batch, reward_batch, action_batch, _ = runner.sample(self.mc_samples)
        perf = TrajectoriesSampler.cumulative_reward(reward_batch)

        # plot info in the log
        log_writer.add_system_parameters(self.env.parameters_dict(), step=self.it)

        log_writer.add_expected_return(perf, step=self.it)

        control_perf = self.env.control_perf(state_batch, action_batch, dist_batch, reward_batch)
        log_writer.add_control_performance(control_perf, step=self.it)

        # next iteration
        self.it += 1

        # if verbose, do the print in the terminal
        if VERY_VERBOSE:
            print(f"Iteration {self.get_it_nb()} : performance = {perf}")

        return -perf

    def _fit_if_required(self, sys, agent):
        policy_algo = None
        if self.policy_algo is not None:
            # the agent has to be fitted
            if isinstance(self.policy_algo, dict):
                policy_algo = initialize.get_algo(sys, agent, initialize=self.init_algo, **self.policy_algo)
            else:
                policy_algo = self.policy_algo(sys, agent)
                policy_algo.initialize(sys, agent, **self.init_algo)

            policy_algo.fit()

        return policy_algo

    def get_it_nb(self):
        return self.it


class MathProg(MathProgBase):

    def __init__(self, env, agent):
        super(MathProg, self).__init__(env=env, agent=agent)

    def fit(self, log_writer=None, device=None):
        # get the optimal parameters
        feasible_points = [tuple(v) for v in self.env.get_feasible_set().values()]
        initial_point = [np.random.uniform(*p) for p in feasible_points]

        self.it = 0
        self.res = scp.minimize(fun=self._get_policy,
                                x0=initial_point,
                                bounds=feasible_points,
                                args=(log_writer,),
                                **self.optim_param)

        # fit the policy with these parameters
        self.agent.set_system(torch.tensor([self.res.x], dtype=torch.float32))
        self._fit_if_required(self.env, self.agent)

        if VERBOSE:
            print(f"Number of iteration : {self.get_it_nb()}")
            print(f"Optimal parameters : {self.res.x}")


class MathProgBrute(MathProgBase):

    def __init__(self, env, agent):
        super(MathProgBrute, self).__init__(env=env, agent=agent)

    def fit(self, log_writer=None, device=None):
        # get the optimal parameters
        feasible_points = [tuple(v) for v in self.env.get_feasible_set().values()]
        self.it = 0

        self.res = scp.brute(
            func=self._get_policy,
            ranges=feasible_points,
            finish=None,
            args=(log_writer,),
            **self.optim_param)

        # fit the policy with these parameters
        self.agent.set_system(torch.tensor([self.res], dtype=torch.float32))
        self._fit_if_required(self.env, self.agent)

        if VERBOSE:
            print(f"Number of iteration : {self.get_it_nb()}")
            print(f"Optimal parameters : {self.res}")


class MathProgGlobal(MathProgBase):

    def __init__(self, env, agent):
        super(MathProgGlobal, self).__init__(env=env, agent=agent)

    def fit(self, log_writer=None, device=None):
        feasible_points = [tuple(v) for v in self.env.get_feasible_set().values()]

        self.it = 0
        self.res = scp.dual_annealing(
            func=self._get_policy,
            bounds=feasible_points,
            args=(log_writer,),
            **self.optim_param)

        # fit the policy with these parameters
        self.agent.set_system(torch.tensor([self.res.x], dtype=torch.float32))
        self._fit_if_required(self.env, self.agent)

        if VERBOSE:
            print(f"Number of iteration : {self.get_it_nb()}")
            print(f"Optimal parameters : {self.res.x}")
