from system.Wrappers.base import SystemWrapper


class ActionClipWrapper(SystemWrapper):
    """ scale the actions played in the system """
    def __init__(self, sys, min=None, max=None):
        super(ActionClipWrapper, self).__init__(sys)

        self.min = min
        self.max = max

    def clip_action(self, action):
        """ scale the action action """
        return action.clip(**{name: val for name, val in zip(["min", "max"], [self.min, self.max]) if val is not None})

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(states, self.clip_action(actions), disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.sys.dynamics(states, self.clip_action(actions), disturbances)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(states, self.clip_action(actions))

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        return self.sys.control_perf(states, self.clip_action(actions), disturbances, rewards)

    def render(self, states, actions, dist, rewards, num_trj):
        return self.sys.render(states, self.clip_action(actions), dist, rewards, num_trj)

    @property
    def unwrapped(self):
        return self.sys.unwrapped
