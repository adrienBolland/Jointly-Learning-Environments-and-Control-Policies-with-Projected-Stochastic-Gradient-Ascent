from system.base import System


class SystemWrapper(System):

    def __init__(self, sys, *args, **kwargs):
        super(SystemWrapper, self).__init__(horizon=sys.horizon,
                                            device=sys.device,
                                            feasible_set=sys.feasible_set)
        self.sys = sys

    @classmethod
    def class_name(cls):
        return cls.__name__

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(states, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.sys.dynamics(states, actions, disturbances)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(states, actions)

    def render(self, states, actions, dist, rewards, num_trj):
        return self.sys.render(states, actions, dist, rewards, num_trj)

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        return self.sys.initial_state(number_trajectories)

    def project_parameters(self):
        """ performs parameters projection """
        return self.sys.project_parameters()

    def set_parameters(self, parameters):
        """ set the parameters to fixed values """
        return self.sys.set_parameters(parameters)

    def parameters_dict(self):
        """ returns a dictionary mapping the parameters' name to their values """
        return self.sys.parameters_dict()

    def get_parameters(self):
        """ returns the parameter vector """
        return self.sys.get_parameters()

    def get_feasible_set(self):
        """ returns the set of feasible values """
        return self.sys.get_feasible_set()

    def control_perf(self, states, actions, disturbances, rewards):
        """Evaluates the performance of the controller only"""
        return self.sys.control_perf(states, actions, disturbances, rewards)

    def to_gym(self):
        return self.sys.to_gym()

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        return self.sys.to(device)

    @property
    def horizon(self):
        """ horizon """
        return self.sys.horizon

    @property
    def action_space(self):
        """ type of action space """
        return self.sys.action_space

    @property
    def observation_space(self):
        """ type of observation space """
        return self.sys.observation_space

    @property
    def parameter_space(self):
        """ type of parameter space """
        return self.sys.parameter_space

    @property
    def differentiable(self):
        return self.sys.differentiable

    @property
    def unwrapped(self):
        return self.sys.unwrapped
