import torch
from torch import nn
from torch.distributions import Normal

import numpy as np

import matplotlib.pyplot as plt

from gym.spaces import Box

from system.base import System


class ComplexMGSystem(System):

    def __init__(self, horizon, dem_size, power_rating, charge_eff, discharge_eff,
                 bat_cost, pv_cost, gen_cost, inv_rate, inv_years, fuel_price, ramp_up_cost, ramp_down_cost,
                 bat_maintenance_cost, pv_maintenance_cost, gen_maintenance_cost,
                 load_curtail_price, load_shed_price, feasible_set=None, device="cpu"):
        super(ComplexMGSystem, self).__init__(horizon=horizon, device=device, feasible_set=feasible_set)
        """ system definition """
        self._observation_space = Box(low=-float('inf'), high=float('inf'), shape=(5,))
        self._action_space = Box(low=-float('inf'), high=float('inf'), shape=(2,))
        self._parameter_space = Box(low=-float('inf'), high=float('inf'), shape=(3,))

        # system parameters
        self.power_rating = torch.tensor([power_rating], dtype=torch.float32, device=self.device)
        self.charge_eff = torch.tensor([charge_eff], dtype=torch.float32, device=self.device)
        self.discharge_eff = torch.tensor([discharge_eff], dtype=torch.float32, device=self.device)
        self.bat_cost = torch.tensor([bat_cost], dtype=torch.float32, device=self.device)
        self.pv_cost = torch.tensor([pv_cost], dtype=torch.float32, device=self.device)
        self.gen_cost = torch.tensor([gen_cost], dtype=torch.float32, device=self.device)
        self.bat_maintenance_cost = torch.tensor([bat_maintenance_cost], dtype=torch.float32, device=self.device)
        self.pv_maintenance_cost = torch.tensor([pv_maintenance_cost], dtype=torch.float32, device=self.device)
        self.gen_maintenance_cost = torch.tensor([gen_maintenance_cost], dtype=torch.float32, device=self.device)
        self.inv_rate = torch.tensor([inv_rate], dtype=torch.float32, device=self.device)
        self.years = torch.tensor([inv_years], dtype=torch.float32, device=self.device)

        # prices in the reward function
        self.fuel_price = torch.tensor([fuel_price], dtype=torch.float32, device=self.device)
        self.load_shed_price = torch.tensor([load_shed_price], dtype=torch.float32, device=self.device)
        self.load_curtail_price = torch.tensor([load_curtail_price], dtype=torch.float32, device=self.device)
        self.ramp_up_cost = torch.tensor([ramp_up_cost], dtype=torch.float32, device=self.device)
        self.ramp_down_cost = torch.tensor([ramp_down_cost], dtype=torch.float32, device=self.device)

        # size (factor) of the demand
        self.dem_size = torch.tensor([dem_size], dtype=torch.float32, device=self.device)

        # Distribution demand and PV production
        self.pv_avg_prod = torch.tensor([0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00,
                                         0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00,
                                         0.00000001e+00, 4.62232374e-02, 8.89720101e-02, 1.22127062e-01,
                                         1.41992336e-01, 1.49666484e-01, 1.43378674e-01, 1.20629623e-01,
                                         8.71089652e-02, 4.64848134e-02, 1.84307861e-17, 0.00000001e+00,
                                         0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00],
                                        device=self.device)

        self.dem_avg = torch.tensor([0.3457438, 0.32335429, 0.309672, 0.29759948, 0.28587788,
                                     0.27293944, 0.24240862, 0.22680175, 0.23042503, 0.23326265,
                                     0.23884741, 0.24825482, 0.25547133, 0.26739509, 0.27287241,
                                     0.27219202, 0.2706911, 0.29403735, 0.42060912, 0.53479381,
                                     0.5502525, 0.5267475, 0.46403763, 0.39285948], device=self.device)

        self.dist_std = torch.tensor([0.05542831, 0.05022998, 0.0432726, 0.03978419, 0.03952021,
                                      0.03775034, 0.03728352, 0.03621157, 0.04035931, 0.04320152,
                                      0.04408169, 0.04740461, 0.04239965, 0.04087229, 0.04240869,
                                      0.04717433, 0.0436305, 0.04424234, 0.08158905, 0.06022856,
                                      0.0553013, 0.05767294, 0.06095378, 0.05918214], device=self.device)

        """ system parameters """
        self.pv_size = torch.tensor([np.nan])
        self.bat_size = torch.tensor([np.nan])
        self.gen_size = torch.tensor([np.nan])

    def set_parameters(self, parameters):
        self.pv_size = parameters[:, (0,)]
        self.bat_size = parameters[:, (1,)]
        self.gen_size = parameters[:, (2,)]

    def initial_state(self, number_trajectories=1):
        soc = (torch.ones((number_trajectories, 1), device=self.device) * 0.5 * self.bat_size).detach()
        h = torch.empty((number_trajectories, 1), device=self.device).zero_()
        gen_prod = torch.empty((number_trajectories, 1), device=self.device).zero_()

        return self._construct_state(soc, h, gen_prod)

    def _construct_state(self, soc, h, gen_prod):
        avg_pv, avg_dem = self._get_avg_pv_dem(h)
        state = torch.cat((soc, h, gen_prod, avg_pv, avg_dem), dim=-1)
        return state

    def _get_soc_h_gen_prod(self, state):
        soc, h, gen_prod, _, _ = state.split(1, dim=-1)
        return soc, h, gen_prod

    def _get_avg_pv_dem(self, h):
        avg_pv = self.pv_avg_prod[h.long()].clone().detach() * self.pv_size
        avg_dem = self.dem_avg[h.long()].clone().detach() * self.dem_size
        return avg_pv, avg_dem

    def disturbance(self, state, action):
        _, h, _ = self._get_soc_h_gen_prod(state)

        sigma = self.dist_std[h.long()]
        mu = torch.zeros_like(sigma)

        return Normal(mu, sigma)

    def reward(self, state, action, disturbances):
        inv_cost = self._amortization()
        var_cost = self._variable_costs(state, action, disturbances)

        # compute the reward
        reward = -(inv_cost + var_cost) * 8760 / float(self.horizon)

        return reward

    def _amortization(self):
        inv_pv = self.pv_size * self.pv_cost + self.pv_size**2 * self.pv_maintenance_cost
        inv_bat = self.bat_size * self.bat_cost + self.bat_size**2 * self.bat_maintenance_cost
        inv_gen = self.gen_size * self.gen_cost + self.gen_size**2 * self.gen_maintenance_cost
        tot_inv = inv_bat + inv_gen + inv_pv
        amort = tot_inv * (self.inv_rate * (1 + self.inv_rate) ** self.years) / (
                (1 + self.inv_rate) ** self.years - 1) / 8760.
        return amort

    def _variable_costs(self, state, action, disturbances):
        # get the soc and the hour from the state
        soc, h, gen_prod = self._get_soc_h_gen_prod(state)

        # compute the pv-prod and the consumption
        avg_pv, avg_dem = self._get_avg_pv_dem(h)
        c_t, pv_t = avg_dem + disturbances, avg_pv

        # rectify the action
        p_bat, p_gen = self._check_actions(soc, action).split(1, dim=-1)

        # compute the generation variable costs (taken positive)
        gen_dif = p_gen - gen_prod
        ramp_up_cost = torch.clamp(gen_dif, min=0)**2 * self.ramp_up_cost
        ramp_down_cost = torch.clamp(-gen_dif, min=0)**2 * self.ramp_down_cost
        gen_fuel_cost = p_gen * self.fuel_price
        gen_cost = gen_fuel_cost + ramp_up_cost + ramp_down_cost

        # compute shed and curtail costs (taken positive)
        diff = self.power_rating * p_gen + pv_t + p_bat - c_t
        load_shed_cost = torch.clamp(-diff * self.load_shed_price, min=0)
        curt_cost = torch.clamp(diff * self.load_curtail_price, min=0)

        # compute the total cost (taken positive)
        cost = gen_cost + load_shed_cost + curt_cost

        return cost

    def dynamics(self, state, action, disturbances):
        # get the soc and the hour from the state
        soc, h, gen_prod = self._get_soc_h_gen_prod(state)

        # rectify the action
        p_bat, p_gen = self._check_actions(soc, action).split(1, dim=-1)

        # get the new state
        next_soc = self._battery_dynamics(soc, p_bat)
        next_gen_prod = self._gen_dynamics(gen_prod, p_gen)
        next_h = (h + 1) % 24

        return self._construct_state(next_soc, next_h, next_gen_prod)

    def _battery_dynamics(self, soc, p_bat):
        """ Returns the new state of charge of the battery when applying a LEGAL action
        p_bat>0 : discharge
        p_bat<0 : charge
        """
        n_s = soc - torch.clamp(p_bat, max=0) * self.charge_eff - torch.clamp(p_bat, min=0) / self.discharge_eff
        return n_s

    def _gen_dynamics(self, gen_prod, p_gen):
        """ Returns the new generation production """
        gen_prod = p_gen
        return gen_prod

    def _check_actions(self, soc, action):
        """ clip the actions to make them legal in the system """
        # get the discharge of the battery and the generator production
        p_bat, p_gen = action.split(1, dim=-1)

        # the battery can at most be discharged to zero and be charged to its max capacity
        p_bat = torch.min(p_bat, soc)
        p_bat = torch.max(p_bat, -(self.bat_size - soc))

        # the generator can not consume power nor produce more than 'self.gen_size'
        p_gen = torch.min(p_gen, self.gen_size)
        p_gen = torch.max(p_gen, torch.tensor([0.]))

        return torch.cat((p_bat, p_gen), dim=-1)

    def project_parameters(self):
        with torch.no_grad():
            pv_ = torch.clamp(self.pv_size,
                              min=self.feasible_set['pv_size'][0], max=self.feasible_set['pv_size'][1])
            bat_ = torch.clamp(self.bat_size,
                               min=self.feasible_set['bat_size'][0], max=self.feasible_set['bat_size'][1])
            gen_ = torch.clamp(self.gen_size,
                               min=self.feasible_set['gen_size'][0], max=self.feasible_set['gen_size'][1])

        for i in range(self.pv_size.shape[0]):
            nn.init.constant_(self.pv_size[i], pv_[i].item())
            nn.init.constant_(self.bat_size[i], bat_[i].item())
            nn.init.constant_(self.gen_size[i], gen_[i].item())

    def render(self, states, actions, dist, rewards, num_trj):
        soc, _, _ = self._get_soc_h_gen_prod(states)
        actions = self._check_actions(soc[:, :-1, :], actions)
        s = states.detach().to('cpu').numpy()
        a = actions.detach().to('cpu').numpy()
        d = dist.detach().to('cpu').numpy()
        r = rewards.detach().to('cpu').numpy()
        for i in range(min(1, num_trj)):
            f, axs = plt.subplots(4, 1, sharex=True)
            for ax, y, t in zip(axs, [s[i], a[i], d[i], r[i]], ['States', "Actions", "Disturbances", "Rewards"]):
                ax.set_title(t)
                ax.plot(y)
        plt.show()

    def control_perf(self, states, actions, disturbances, rewards):
        """Evaluates the variable costs only"""
        # transpose since the environments are set by batch
        var_cost = self._variable_costs(states[:, :-1, :].transpose(0, 1),
                                        actions.transpose(0, 1),
                                        disturbances.transpose(0, 1))

        # the sum is done along axis 0 as the tensors were transposed
        return -torch.mean(torch.sum(var_cost, dim=0))

    def parameters_dict(self):
        return {'pv_size': self.pv_size.mean().item(),
                'bat_size': self.bat_size.mean().item(),
                'gen_size': self.gen_size.mean().item()}

    def get_parameters(self):
        return torch.cat([self.pv_size, self.bat_size, self.gen_size], dim=-1)

    def to_gym(self):
        raise NotImplementedError

    @property
    def differentiable(self):
        return True
