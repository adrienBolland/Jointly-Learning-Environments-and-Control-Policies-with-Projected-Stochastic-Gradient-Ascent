import torch

from system.Wrappers.base import SystemWrapper


class StationarySpeedScaleWrapper(SystemWrapper):
    """ apply a constant speed corresponding to the speed for a constant altitude """
    def __init__(self, sys):
        super(StationarySpeedScaleWrapper, self).__init__(sys)

    def unscale_action(self, action):
        """ reproduce four times the same action """
        gravity_force = self.sys.unwrapped._get_mass() * self.sys.unwrapped.g
        thrust_factor, _ = self.sys.unwrapped._get_thrust_drag_factors()
        stationary_speed = torch.sqrt(gravity_force / (4 * thrust_factor))

        return action + stationary_speed.view(-1, *[1 for _ in action.shape[1:]])

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(states, self.unscale_action(actions), disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.sys.dynamics(states, self.unscale_action(actions), disturbances)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(states, self.unscale_action(actions))

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        return self.sys.control_perf(states, self.unscale_action(actions), disturbances, rewards)

    def render(self, states, actions, dist, rewards, num_trj):
        return self.sys.render(states, self.unscale_action(actions), dist, rewards, num_trj)

    @property
    def unwrapped(self):
        return self.sys.unwrapped
