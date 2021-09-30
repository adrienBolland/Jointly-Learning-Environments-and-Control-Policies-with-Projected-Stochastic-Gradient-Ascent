from abc import ABC, abstractmethod

import torch
from torch import nn

import math
import numpy as np

from gym.spaces import Box

from system.base import System


class BaseDrone(System, ABC):

    def __init__(self, horizon, discrete_time, euler_time, initial_radius=0., initial=(0., 0., 0.),
                 feasible_set=None, device="cpu"):
        super(BaseDrone, self).__init__(horizon=horizon, device=device, feasible_set=feasible_set)
        # spaces
        self._observation_space = Box(low=-float("inf"), high=float("inf"), shape=(12,))
        self._action_space = Box(low=-float("inf"), high=float("inf"), shape=(4,))
        self._parameter_space = Box(low=-float("inf"), high=float("inf"), shape=(8,))

        # target equilibrium
        self.initial = torch.tensor([initial], device=self.device)

        # radius of initial position
        self.initial_radius = torch.tensor([initial_radius, initial_radius, initial_radius], device=self.device)

        # Euler integration
        self.euler_time = euler_time
        self.discrete_time = discrete_time

        # physical constants
        self.g = 9.81
        self.rho = 900.
        self.rho_air = 1.225
        self.pi = np.pi

        self.c_b = 1.
        self.c_d = 1.

        self.parameters = torch.tensor([np.nan] * 8, device=self.device)

    def project_parameters(self):
        """ performs parameters projection """
        for i, (min_val, max_val) in enumerate(self.get_feasible_set().values()):
            with torch.no_grad():
                val_proj = self.parameters[:, i].clamp(min_val, max_val)

            for b in range(self.parameters.shape[0]):
                nn.init.constant_(self.parameters[b, i], val_proj[b].item())

    def set_parameters(self, parameters):
        """ set the parameters to fixed values """
        self.parameters = parameters

    def initial_state(self, number_trajectories):
        """ samples "number_trajectories" initial states from P_0
         returns a tensor of shape ("number_trajectories", |S|) """
        xyz = 2 * torch.rand(number_trajectories, 3, device=self.device) * self.initial_radius - self.initial
        state_not_xyz = torch.zeros((number_trajectories, self.observation_space.shape[0] - 3), device=self.device)

        return torch.cat([state_not_xyz, xyz], dim=-1)

    @abstractmethod
    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        raise NotImplementedError

    def _distance_omega(self, states, actions, disturbances):
        omega_1, omega_2, omega_3, omega_4 = actions.split(1, dim=-1)
        _, _, _, _, omega_1n, omega_2n, omega_3n, omega_4n = self._get_system_parameters()
        distance_omega = ((omega_1 - omega_1n).pow(2) + (omega_2 - omega_2n).pow(2)
                          + (omega_3 - omega_3n).pow(2) + (omega_4 - omega_4n).pow(2))

        return distance_omega

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        s_next = states

        for _ in range(math.ceil(self.discrete_time / self.euler_time)):
            # linear evolution
            s_next = s_next + self._derivative(s_next, actions, disturbances) * self.euler_time

            # Euler angles are bounded (with a circular evolution)
            phi, theta, psi, p, q, r, u, v, w, x, y, z = s_next.split(1, dim=-1)

            # phi \in ] -pi, pi]
            phi = self.circular_bound(phi, self.pi)

            # theta \in ] -pi/2, pi/2]
            # theta = self.circular_bound(theta, self.pi)

            # phi \in ] -pi, pi]
            psi = self.circular_bound(psi, self.pi)

            s_next = torch.cat([phi, theta, psi, p, q, r, u, v, w, x, y, z], dim=-1)

        return s_next

    def _derivative(self, states, actions, disturbances):
        """ computes the derivative of the state variables with the transition model """
        # get the inertia and the mass
        i_x, i_y, i_z = self._get_inertia()
        m = self._get_mass()

        # get the motor forces on the system
        f_t, tau_x, tau_y, tau_z = self._get_motor_forces(actions)

        # get the state components
        phi, theta, psi, p, q, r, u, v, w, x, y, z = states.split(1, dim=-1)

        # get the wind disturbances and put the forces to the local frame
        glob_f_wx, glob_f_wy, glob_f_wz, tau_wx, tau_wy, tau_wz = (self.disturbance_to_force(states,
                                                                                             actions,
                                                                                             disturbances)
                                                                   .split(1, dim=-1))

        f_wx = ((torch.cos(theta) * torch.cos(psi)) * glob_f_wx
                + (torch.cos(theta) * torch.sin(psi)) * glob_f_wy
                - torch.sin(psi) * glob_f_wz)
        f_wy = ((torch.sin(phi) * torch.sin(theta) * torch.cos(psi) - torch.cos(phi) * torch.sin(psi)) * glob_f_wx
                + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi) + torch.cos(phi) * torch.cos(psi)) * glob_f_wy
                + torch.sin(phi) * torch.cos(theta) * glob_f_wz)
        f_wz = ((torch.cos(phi) * torch.sin(theta) * torch.cos(psi) + torch.sin(phi) * torch.sin(psi)) * glob_f_wx
                + (torch.cos(phi) * torch.sin(theta) * torch.sin(psi) - torch.sin(phi) * torch.cos(psi)) * glob_f_wy
                + torch.cos(phi) * torch.cos(theta) * glob_f_wz)

        # compute the state derivative with newton's formula and the change of coordinates
        dot_p = (i_y - i_z) / i_x * r * q + (tau_x + tau_wx) / i_x
        dot_q = (i_z - i_x) / i_y * p * r + (tau_y + tau_wy) / i_y
        dot_r = (i_x - i_y) / i_z * p * q + (tau_z + tau_wz) / i_z

        dot_u = r * v - q * w - self.g * torch.sin(theta) + f_wx / m
        dot_v = p * w - r * u + self.g * torch.sin(phi) * torch.cos(theta) + f_wy / m
        dot_w = q * u - p * v + self.g * torch.cos(phi) * torch.cos(theta) + (f_wz - f_t) / m

        dot_x = ((torch.cos(theta) * torch.cos(psi)) * u
                 + (torch.sin(phi) * torch.sin(theta) * torch.cos(psi) - torch.cos(phi) * torch.sin(psi)) * v
                 + (torch.cos(phi) * torch.sin(theta) * torch.cos(psi) + torch.sin(phi) * torch.sin(psi)) * w)
        dot_y = ((torch.cos(theta) * torch.sin(psi)) * u
                 + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi) + torch.cos(phi) * torch.cos(psi)) * v
                 + (torch.cos(phi) * torch.sin(theta) * torch.sin(psi) - torch.sin(phi) * torch.cos(psi)) * w)
        dot_z = (-torch.sin(psi) * u
                 + torch.sin(phi) * torch.cos(theta) * v
                 + torch.cos(phi) * torch.cos(theta) * w)

        dot_phi = p + torch.sin(phi) * torch.tan(theta) * q + torch.cos(phi) * torch.tan(theta) * r
        dot_theta = torch.cos(phi) * q - torch.sin(phi) * r
        dot_psi = (torch.sin(phi) / torch.cos(theta)) * q + (torch.cos(phi) / torch.cos(theta)) * r

        return torch.cat([dot_phi, dot_theta, dot_psi, dot_p, dot_q, dot_r, dot_u, dot_v, dot_w, dot_x, dot_y, dot_z],
                         dim=-1)

    def _get_inertia(self):
        l, r_b, t, w, _, _, _, _ = self._get_system_parameters()

        i_x = (1 / 6.) * self.rho * l * t * (4 * l**2 + t**2)
        i_y = i_x.clone()
        i_z = (1 / 3.) * self.rho * l * w * (4 * l**2 + w**2) - (1 / 6.) * self.rho * w**4

        return i_x, i_y, i_z

    def _get_mass(self):
        l, r_b, t, w, _, _, _, _ = self._get_system_parameters()
        m = self.rho * t * w * (2 * (2 * l - w) + w)

        return m

    def _get_motor_forces(self, actions):
        # thrust (b) and drag (d) factors
        b, d = self._get_thrust_drag_factors()

        # get l
        l, _, _, _, _, _, _, _ = self._get_system_parameters()

        # actions correspond to the angular speeds of the rotors
        omega_1, omega_2, omega_3, omega_4 = actions.split(1, dim=-1)

        # force from the speed
        f_t = b * (omega_1**2 + omega_2**2 + omega_3**2 + omega_4**2)

        # torques from the speed
        tau_x = b * l * (omega_3**2 - omega_1**2)
        tau_y = b * l * (omega_4**2 - omega_2**2)
        tau_z = d * (omega_2**2 + omega_4**2 - omega_1**2 - omega_3**2)

        return f_t, tau_x, tau_y, tau_z

    def _get_thrust_drag_factors(self):
        # thrust (b) and drag (d) factors
        _, r_b, _, _, _, _, _, _ = self._get_system_parameters()
        area = self.pi * r_b**2
        b = 0.5 * self.rho_air * self.c_b * area * r_b**2
        d = 0.5 * self.rho_air * self.c_d * area * r_b**2

        return b, d

    def _get_system_parameters(self):
        phi_0, phi_1, phi_2, phi_3, omega_1, omega_2, omega_3, omega_4 = self.parameters.split(1, dim=-1)
        l, r_b, t, w = phi_0 + phi_1, phi_1, phi_2, phi_3

        return l, r_b, t, w, omega_1, omega_2, omega_3, omega_4

    @staticmethod
    def circular_bound(val, bound):
        # sign depends on how many circle were performed
        odd_overflow = (torch.floor(val / bound) - (val < 0.).float()).fmod(2)

        return val.fmod(bound) - odd_overflow * bound

    @abstractmethod
    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        raise NotImplementedError

    def disturbance_to_force(self, states, actions, dist):
        """ transforms a disturbance into a force and a torque applied on the drone """
        return dist

    @abstractmethod
    def render(self, states, actions, dist, rewards, num_trj):
        raise NotImplementedError

    @abstractmethod
    def control_perf(self, states, actions, disturbances, rewards):
        raise NotImplementedError

    def parameters_dict(self):
        """ returns a dictionary mapping the parameters" name to their values """
        return {name: self.get_parameters()[:, i].mean().item() for i, name in enumerate(self.get_feasible_set())}

    def get_feasible_set(self):
        """ returns the set of feasible values """
        return {"arm": self.feasible_set["arm"],
                "radius": self.feasible_set["radius"],
                "thickness": self.feasible_set["thickness"],
                "width": self.feasible_set["width"],
                **{f"speed-{i}": self.feasible_set["speed"] for i in range(self.parameter_space.shape[0] - 4)}}

    def get_parameters(self):
        """ returns the parameter vector """
        return self.parameters

    def to_gym(self):
        """ builds a gym environment """
        raise NotImplementedError
