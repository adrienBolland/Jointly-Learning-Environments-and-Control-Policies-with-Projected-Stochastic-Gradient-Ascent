import torch

from system.Wrappers.base import SystemWrapper


class FixedSysWrapper(SystemWrapper):

    def __init__(self, sys, parameters):
        super(FixedSysWrapper, self).__init__(sys)
        sys.set_parameters(torch.tensor([parameters]))
