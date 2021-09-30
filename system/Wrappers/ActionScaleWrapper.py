from system.Wrappers.base import SystemWrapper


class ActionScaleWrapper(SystemWrapper):
    """ scale the actions played in the system """
    def __init__(self, sys, loc, scale):
        super(ActionScaleWrapper, self).__init__(sys)

        self.loc = loc
        self.scale = scale

    def unscale_action(self, action):
        """ scale the action action """
        return (action - self.loc) / self.scale

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
