import torch

from system.Wrappers.base import SystemWrapper


class OHEWrapper(SystemWrapper):
    def __init__(self, sys):
        super(OHEWrapper, self).__init__(sys)

        self.num_to_ohe_tensor = torch.FloatTensor(1, self.sys.action_space.n).zero_()
        self.ohe_to_num_tensor = torch.arange(start=0, end=self.sys.action_space.n).reshape(-1, 1).float()

    def num_to_ohe(self, num_action):
        return self.num_to_ohe_tensor.repeat(num_action.shape[0], 1).scatter(1, num_action, 1)

    def ohe_to_num(self, ohe_action):
        return torch.matmul(ohe_action, self.ohe_to_num_tensor).long()

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(states, self.ohe_to_num(actions), disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.sys.dynamics(states, self.ohe_to_num(actions), disturbances)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(states, self.ohe_to_num(actions))

    def control_perf(self, states, actions, disturbances, rewards):
        """ evaluates the performance of the controller only """
        return self.sys.control_perf(states, self.ohe_to_num(actions), disturbances, rewards)

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.num_to_ohe_tensor = self.num_to_ohe_tensor.to(device)
        self.ohe_to_num_tensor = self.ohe_to_num_tensor.to(device)
        self.sys.to(device)

    @property
    def unwrapped(self):
        return self.sys.unwrapped
