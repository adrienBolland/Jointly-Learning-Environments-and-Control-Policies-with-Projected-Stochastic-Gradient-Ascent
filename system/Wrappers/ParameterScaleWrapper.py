import torch
from system.Wrappers.base import SystemWrapper


class ParameterScaleWrapper(SystemWrapper):
    def __init__(self, sys, loc, scale):
        super(ParameterScaleWrapper, self).__init__(sys)
        self.loc = torch.tensor([loc])
        self.scale = torch.tensor([scale])

    def scale_parameter(self, parameter):
        return (parameter - self.loc) / self.scale

    def unscale_parameter(self, scaled_parameter):
        return (scaled_parameter * self.scale) + self.loc

    def set_parameters(self, parameters):
        """ set the parameters to fixed values """
        return self.sys.set_parameters(self.unscale_parameter(parameters))

    def parameters_dict(self):
        return {name: p for name, p in zip(self.sys.parameters_dict(), self.get_parameters().flatten().tolist())}

    def get_parameters(self):
        """ returns the parameter vector """
        return self.scale_parameter(self.sys.get_parameters())

    def get_feasible_set(self):
        """ returns the set of feasible values """
        d = self.sys.get_feasible_set()
        return {name: [(v - self.loc[0, i].item()) / self.scale[0, i].item() for v in d[name]]
                for i, name in enumerate(d)}

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.loc = self.loc.to(device)
        self.scale = self.scale.to(device)
        self.sys.to(device)

    @property
    def unwrapped(self):
        return self.sys.unwrapped
