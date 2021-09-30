import numpy as np

import torch
from torch import nn
from torch.distributions import Normal

from gym.spaces import Discrete, Box

import matplotlib.pyplot as plt

from system.base import System


class MSDSystem(System):
    def __init__(self, horizon, equilibrium, actions_value, target_parameters_reward, cost_omega_zeta, accuracy,
                 actions_discretization, position_interval, speed_interval, feasible_set, nb_phi=3, device="cpu"):
        super(MSDSystem, self).__init__(horizon=horizon, device=device, feasible_set=feasible_set)

        """ System definition """
        # state space
        self._observation_space = Box(low=-float('inf'), high=float('inf'), shape=(2,))

        # initial position and speed intervals for the initial distribution over states
        self.position_interval = position_interval
        self.speed_interval = speed_interval

        # action space
        self._action_space = Discrete(n=len(actions_value))
        self.actions_value = torch.tensor(actions_value, device=self.device)

        # equilibrium position in the reward
        self.equilibrium = equilibrium

        # accuracy in the reward (lambda)
        self.accuracy = accuracy

        # target constant parameters in the reward
        self.target_parameters = torch.tensor([target_parameters_reward], device=self.device)

        # cost of omega and zeta in the reward
        self.cost_omega_zeta = torch.tensor([cost_omega_zeta], device=self.device)

        # time discretization of the system
        self.actions_discretization = actions_discretization

        """ System parameters """
        # feasible set of parameters
        self.omega_interval = feasible_set["omega_interval"]
        self.zeta_interval = feasible_set["zeta_interval"]
        self.phi_interval = feasible_set["phi_interval"]

        self._parameter_space = Box(low=-float('inf'), high=float('inf'), shape=(2 + len(target_parameters_reward),))

        # parameter values
        self.nb_phi = nb_phi
        self.phi_param = torch.tensor([[np.nan] * nb_phi])
        self.omega_zeta_param = torch.tensor([[np.nan, np.nan]])

    def set_parameters(self, parameters):
        self.omega_zeta_param = parameters[:, :2]
        self.phi_param = parameters[:, 2:]

    def project_parameters(self):
        with torch.no_grad():
            omega = self.omega_zeta_param[:, 0].clamp(self.omega_interval[0], self.omega_interval[1])
            zeta = self.omega_zeta_param[:, 1].clamp(self.zeta_interval[0], self.zeta_interval[1])
            phi = self.phi_param.clamp(self.phi_interval[0], self.phi_interval[1])

        for b in range(self.omega_zeta_param.shape[0]):
            nn.init.constant_(self.omega_zeta_param[b, 0], omega[b].item())
            nn.init.constant_(self.omega_zeta_param[b, 1], zeta[b].item())

            for i in range(self.phi_param.shape[1]):
                nn.init.constant_(self.phi_param[b, i], phi[b, i].item())

    def dynamics(self, previous_states, actions, disturbances):
        return MSDSystem.f(previous_states[:, 0:1],
                           previous_states[:, 1:2],
                           self.actions_value[actions] + disturbances,
                           self.actions_discretization,
                           self.omega_zeta_param[:, (0,)],
                           self.omega_zeta_param[:, (1,)])

    @staticmethod
    def f(x_0, v_0, a, t, omega, zeta):
        # new state
        position = torch.empty(x_0.shape)
        speed = torch.empty(v_0.shape)

        # repeat if a single parameter was used
        omega = torch.repeat_interleave(omega, x_0.shape[0] // omega.shape[0], dim=0)
        zeta = torch.repeat_interleave(zeta, x_0.shape[0] // zeta.shape[0], dim=0)

        # indices
        indices = torch.arange(zeta.shape[0]).view(-1, 1)

        # zeta > 1.
        id = indices[zeta > 1.]
        position[id, :], speed[id, :] = MSDSystem._f_z_greater_1(x_0.index_select(0, id), v_0.index_select(0, id),
                                                                 a.index_select(0, id), t, omega.index_select(0, id),
                                                                 zeta.index_select(0, id))

        # zeta = 1.
        id = indices[zeta == 1.]
        position[id, :], speed[id, :] = MSDSystem._f_z_is_1(x_0.index_select(0, id), v_0.index_select(0, id),
                                                            a.index_select(0, id), t, omega.index_select(0, id),
                                                            zeta.index_select(0, id))

        # zeta < 1.
        id = indices[zeta < 1.]
        position[id, :], speed[id, :] = MSDSystem._f_z_lesser_1(x_0.index_select(0, id), v_0.index_select(0, id),
                                                                a.index_select(0, id), t, omega.index_select(0, id),
                                                                zeta.index_select(0, id))

        return torch.cat((position, speed), 1)

    @staticmethod
    def _f_z_greater_1(x_0, v_0, a, t, omega, zeta):
        omega_t = omega * t
        action = a / omega.pow(2)

        root = (zeta.pow(2) - 1).sqrt()

        position = ((-zeta * omega_t).exp()
                    * ((x_0 - action) * (root * omega_t).cosh()
                       + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sinh())
                    + action)

        speed = ((-zeta * omega * (-zeta * omega_t).exp()
                  * ((x_0 - action) * (root * omega_t).cosh()
                     + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sinh()))
                 + ((-zeta * omega_t).exp()
                    * ((x_0 - action) * root * omega * (root * omega_t).sinh()
                       + (v_0 + zeta * (x_0 - action) * omega) * (root * omega_t).cosh())))

        return position, speed

    @staticmethod
    def _f_z_is_1(x_0, v_0, a, t, omega, zeta):
        omega_t = omega * t
        action = a / omega.pow(2)

        position = ((x_0 - action) + (v_0 + omega * (x_0 - action)) * t) * (- omega_t).exp() + action

        speed = (((x_0 - action) + (v_0 + omega * (x_0 - action)) * t) * (-omega) * (- omega_t).exp()
                 + (v_0 + omega * (x_0 - action)) * (- omega_t).exp())

        return position, speed

    @staticmethod
    def _f_z_lesser_1(x_0, v_0, a, t, omega, zeta):
        omega_t = omega * t
        action = a / omega.pow(2)

        root = (1 - zeta.pow(2)).sqrt()

        position = ((-zeta * omega_t).exp()
                    * ((x_0 - action) * (root * omega_t).cos()
                       + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sin())
                    + action)

        speed = ((-zeta * omega * (-zeta * omega_t).exp()
                  * ((x_0 - action) * (root * omega_t).cos()
                     + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sin()))
                 + ((-zeta * omega_t).exp()
                    * (- (x_0 - action) * root * omega * (root * omega_t).sin()
                       + (v_0 + zeta * (x_0 - action) * omega) * (root * omega_t).cos())))

        return position, speed

    def initial_state(self, number_trajectories=1):
        p = torch.empty((number_trajectories, 1), device=self.device).uniform_(self.position_interval[0],
                                                                               self.position_interval[1])
        s = torch.empty((number_trajectories, 1), device=self.device).uniform_(self.speed_interval[0],
                                                                               self.speed_interval[1])
        return torch.cat((p, s), dim=1)

    def reward(self, states, actions, disturbances):
        position_error = torch.abs(states[:, :1] - self.equilibrium)
        parameters_error = torch.prod((self.phi_param - self.target_parameters).pow(2), dim=-1, keepdim=True)
        omega_zeta_error = torch.sum((self.omega_zeta_param - self.cost_omega_zeta).pow(2), dim=-1, keepdim=True)

        error = (position_error / self.accuracy + omega_zeta_error + parameters_error)

        error = torch.exp(- error)

        return error

    def disturbance(self, states, actions):
        numerical_actions = self.actions_value[actions]
        positions, speeds = states[:, 0:1], states[:, 1:2]

        mu = positions
        sigma = 0.1 * numerical_actions.abs() + speeds.abs() + 10 ** -6

        return Normal(mu, sigma)

    def control_perf(self, states, actions, disturbances, rewards):
        return torch.mean(torch.abs(states[:, :, 0] - self.equilibrium))

    def render(self, states, actions, dist, rewards, num_trj):
        """ graphical view """
        plt.figure()
        plt.plot(states[0, :, 0].tolist())
        plt.figure()
        plt.plot(actions[0, :, 0].tolist())
        plt.show()
        pass

    @property
    def differentiable(self):
        return True

    def parameters_dict(self):
        return {'omega': self.omega_zeta_param[:, 0].mean().item(),
                'zeta': self.omega_zeta_param[:, 1].mean().item(),
                **{f'phi-{i}': self.phi_param[:, i].mean().item() for i in range(self.nb_phi)}}

    def get_parameters(self):
        """ returns the parameter vector """
        return torch.cat([self.omega_zeta_param, self.phi_param], dim=1)

    def get_feasible_set(self):
        """ returns the set of feasible values """
        return {'omega': self.omega_interval,
                'zeta': self.zeta_interval,
                **{f'phi-{i}': self.phi_interval for i in range(self.nb_phi)}}

    @staticmethod
    def states_repr():
        return {0: {"title": '$x_{t}$ - mass position [m]',
                    "name": 'position'},
                1: {"title": '$s_{t}$ - mass speed [m/s]',
                    "name": 'speed'}}
