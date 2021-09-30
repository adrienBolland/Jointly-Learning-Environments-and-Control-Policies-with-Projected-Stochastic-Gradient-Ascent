from system.Wrappers.base import SystemWrapper


class StateClipWrapper(SystemWrapper):
    def __init__(self, sys, min, max):
        super(StateClipWrapper, self).__init__(sys)
        self.min = min
        self.max = max

    def scale_state(self, unscaled_state):
        return unscaled_state.clamp(min=self.min, max=self.max)

    def unscale_state(self, scaled_state):
        # the transformation is not bijective
        return scaled_state

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        return self.scale_state(self.sys.initial_state(number_trajectories))

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(self.unscale_state(states), actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.scale_state(self.sys.dynamics(self.unscale_state(states), actions, disturbances))

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(self.unscale_state(states), actions)

    def render(self, states, actions, dist, rewards, num_trj):
        """ render of a batch """
        return self.sys.render(self.unscale_state(states), actions, dist, rewards, num_trj)

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        return self.sys.control_perf(self.unscale_state(states), actions, disturbances, rewards)

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.sys.to(device)

    @property
    def unwrapped(self):
        return self.sys.unwrapped
