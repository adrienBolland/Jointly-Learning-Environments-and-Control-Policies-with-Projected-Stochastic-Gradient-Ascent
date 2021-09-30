import torch
import os

from agent.base import BaseAgent


class CMGRuleAgent(BaseAgent):
    def __init__(self):
        super(CMGRuleAgent, self).__init__()

        self.env_param = None

        self.bat_size = None
        self.gen_size = None

    def get_system(self):
        return self.env_param

    def set_system(self, parameters):
        self.env_param = parameters

        self.bat_size = parameters[0, (1,)]
        self.gen_size = parameters[0, (2,)]

    def get_action(self, observation):
        soc, _, _, avg_pv, avg_dem = observation.split(1, dim=-1)

        expected_shortage = avg_dem - avg_pv

        battery_discharge = torch.min(expected_shortage, soc)
        battery_discharge = torch.max(battery_discharge, -(self.bat_size - soc))

        gen_prod = expected_shortage - battery_discharge
        gen_prod = torch.min(gen_prod, self.gen_size)
        gen_prod = torch.max(gen_prod, torch.tensor([0.]))

        return torch.cat((battery_discharge, gen_prod), dim=-1)

    def initialize(self, env, **kwargs):
        self.env_param = env.get_parameters()

        self.bat_size = env.unwrapped.bat_size
        self.gen_size = env.unwrapped.gen_size
        return self

    def save(self, path):
        """Save models"""
        path = os.path.join(path, f'agent-save/investment-pol/investment')

        dir, file = os.path.split(path)
        if dir != '':
            os.makedirs(dir, exist_ok=True)  # required if directory not created yet

        torch.save(self.env_param, path)

    def load(self, path):
        """Load models"""
        path = os.path.join(path, f'agent-save/investment-pol/investment')
        self.set_system(torch.load(path))


class CMGRuleAgentBL(BaseAgent):
    """ agent where the generator is used for providing a constant base load"""
    def __init__(self):
        super(CMGRuleAgentBL, self).__init__()

        self.env_param = None

        self.bat_size = None
        self.gen_size = None

    def get_system(self):
        return self.env_param

    def set_system(self, parameters):
        self.env_param = parameters

        self.bat_size = parameters[0, (1,)]
        self.gen_size = parameters[0, (2,)]

    def get_action(self, observation):
        soc, _, _, avg_pv, avg_dem = observation.split(1, dim=-1)

        gen_prod = torch.empty((soc.shape[0], 1)).fill_(self.gen_size.item())

        expected_shortage = avg_dem - avg_pv - gen_prod

        battery_discharge = torch.min(expected_shortage, soc)
        battery_discharge = torch.max(battery_discharge, -(self.bat_size - soc))

        return torch.cat((battery_discharge, gen_prod), dim=-1)

    def initialize(self, env, **kwargs):
        self.env_param = env.get_parameters()

        self.bat_size = env.unwrapped.bat_size
        self.gen_size = env.unwrapped.gen_size
        return self

    def save(self, path):
        """Save models"""
        path = os.path.join(path, f'agent-save/investment-pol/investment')

        dir, file = os.path.split(path)
        if dir != '':
            os.makedirs(dir, exist_ok=True)  # required if directory not created yet

        torch.save(self.env_param, path)

    def load(self, path):
        """Load models"""
        path = os.path.join(path, f'agent-save/investment-pol/investment')
        self.set_system(torch.load(path))
