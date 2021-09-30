from system.Wrappers.base import SystemWrapper


class RewardClipWrapper(SystemWrapper):
    def __init__(self, sys, min=0., max=0.):
        super(RewardClipWrapper, self).__init__(sys)
        self.min = min
        self.max = max

    def scale_rew(self, reward):
        return reward.clamp(min=self.min, max=self.max)

    def unscale_rew(self, scaled_rew):
        # the transformation is not bijective
        return scaled_rew

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.scale_rew(self.sys.reward(states, actions, disturbances))

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        return self.sys.control_perf(states, actions, disturbances, self.unscale_rew(rewards))

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.sys.to(device)

    @property
    def unwrapped(self):
        return self.sys.unwrapped
