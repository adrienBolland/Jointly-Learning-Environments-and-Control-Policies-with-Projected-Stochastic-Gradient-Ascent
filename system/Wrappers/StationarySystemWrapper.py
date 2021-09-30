from copy import deepcopy

import torch
from system.Wrappers.base import SystemWrapper


class StationarySystemWrapper(SystemWrapper):
    def __init__(self, sys):
        super(StationarySystemWrapper, self).__init__(sys)

    def time_state(self, time_indep_state, t):
        return torch.cat((time_indep_state, t / float(self.horizon)), dim=-1)

    def no_time_state(self, time_dep_state):
        time_indep_state, t = time_dep_state.split((time_dep_state.shape[-1] - 1, 1), dim=-1)
        return time_indep_state, t * float(self.horizon)

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        time = torch.zeros((number_trajectories, 1), device=self.device)
        return self.time_state(self.sys.initial_state(number_trajectories), time)

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        time_indep_state, _ = self.no_time_state(states)
        return self.sys.reward(time_indep_state, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        time_indep_state, time = self.no_time_state(states)
        return self.time_state(self.sys.dynamics(time_indep_state, actions, disturbances), time+1)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        time_indep_state, _ = self.no_time_state(states)
        return self.sys.disturbance(time_indep_state, actions)

    def render(self, states, actions, dist, rewards, num_trj):
        """ render of a batch """
        time_indep_state, _ = self.no_time_state(states)
        return self.sys.render(time_indep_state, actions, dist, rewards, num_trj)

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        time_indep_state, _ = self.no_time_state(states)
        return self.sys.control_perf(time_indep_state, actions, disturbances, rewards)

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.device = device
        self.sys.to(device)

    @property
    def observation_space(self):
        """ type of observation space """
        space = deepcopy(self.sys.observation_space)
        space.shape = (space.shape[0] + 1,)
        return space

    @property
    def unwrapped(self):
        return self.sys.unwrapped
