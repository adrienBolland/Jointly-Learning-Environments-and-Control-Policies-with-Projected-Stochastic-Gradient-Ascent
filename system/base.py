from abc import ABC, abstractmethod
from collections import namedtuple

import torch

from system.togym import GymEnv

EnvStep = namedtuple("EnvStep",
                     ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", [])
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class System(ABC):
    """ rem : the object manipulates the actions as outputted by the policy network"""

    def __init__(self, horizon, device="cpu", feasible_set=None):
        super(System, self).__init__()
        self.device = device
        self.feasible_set = feasible_set

        self._horizon = horizon
        self._action_space = None
        self._observation_space = None
        self._parameter_space = None

    @abstractmethod
    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        raise NotImplementedError

    @abstractmethod
    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        raise NotImplementedError

    @abstractmethod
    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        raise NotImplementedError

    def render(self, states, actions, dist, rewards, num_trj):
        """ graphical view """
        pass

    def __call__(self, state, action):
        """ execution of the system (callable) """
        return self.forward(state, action)

    def forward(self, state, action):
        """ execution of the system """

        disturbances = self.disturbance(state, action).sample()

        reward = self.reward(state, action, disturbances)

        next_states = self.dynamics(state, action, disturbances)
        return next_states, disturbances, reward, action

    @abstractmethod
    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        raise NotImplementedError

    @abstractmethod
    def project_parameters(self):
        """ performs parameters projection """
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, parameters):
        """ set the parameters to fixed values """
        raise NotImplementedError

    def control_perf(self, states, actions, disturbances, rewards):
        """Evaluates the performance of the controller only"""
        return 0

    def parameters_dict(self):
        """ returns a dictionary mapping the parameters' name to their values """
        return dict()

    def get_feasible_set(self):
        """ returns the set of feasible values """
        return self.feasible_set

    @abstractmethod
    def get_parameters(self):
        """ returns the parameter vector """
        raise NotImplementedError

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.device = device

        for var_name, var_ptr in vars(self).items():
            if torch.is_tensor(var_ptr):
                vars(self)[var_name] = var_ptr.to(device)

    def to_gym(self):
        """ builds a gym environment """
        return GymEnv(self)

    @property
    def horizon(self):
        """ horizon """
        return self._horizon

    @property
    def action_space(self):
        """ type of action space """
        return self._action_space

    @property
    def observation_space(self):
        """ type of observation space """
        return self._observation_space

    @property
    def parameter_space(self):
        """ type of parameter space """
        return self._parameter_space

    @property
    def spaces(self):
        return EnvSpaces(observation=self.observation_space,
                         action=self.action_space)

    @property
    def differentiable(self):
        return False

    @property
    def unwrapped(self):
        """Completely unwrap this systems
        """
        return self
