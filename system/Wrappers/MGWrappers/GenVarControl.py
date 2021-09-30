import torch

from system.Wrappers.base import SystemWrapper


class GenVarControl(SystemWrapper):
    def __init__(self, sys):
        super(GenVarControl, self).__init__(sys)

    def absolute_to_relative(self, state, action):
        _, _, prev_gen_prod, _, _ = state.split(1, dim=-1)
        bat_discharge, gen_prod = action.split(1, dim=-1)

        return torch.cat([bat_discharge, gen_prod - prev_gen_prod], dim=-1)

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(states, self.absolute_to_relative(states, actions), disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.sys.dynamics(states, self.absolute_to_relative(states, actions), disturbances)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(states, self.absolute_to_relative(states, actions))

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        return self.sys.control_perf(states, self.absolute_to_relative(states[:, :-1, :], actions), disturbances,
                                     rewards)

    def render(self, states, actions, dist, rewards, num_trj):
        return self.sys.render(states, self.absolute_to_relative(states[:, :-1, :], actions), dist, rewards, num_trj)

    @property
    def unwrapped(self):
        return self.sys.unwrapped