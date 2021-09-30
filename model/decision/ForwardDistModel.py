from abc import ABC, abstractmethod

from torch import nn


class ForwardDistModel(nn.Module, ABC):

    def __init__(self, input_size, output_size, layers, act_fun=nn.Tanh):
        super(ForwardDistModel, self).__init__()

        if type(act_fun) is str:
            act_fun = eval(f"nn.{act_fun}")

        self.input_size = input_size
        self.output_size = output_size

        self.layers = []
        for n_neurons in layers:
            # linear layers
            self.layers.append(nn.Linear(input_size, n_neurons))
            self.layers.append(act_fun())

            input_size = n_neurons

        self.layers.append(nn.Linear(input_size, output_size))

        self.net = nn.Sequential(*self.layers)

    @abstractmethod
    def forward(self, x):
        """ returns a distribution """
        raise NotImplementedError

    def reset_parameters(self):

        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.net.apply(weight_reset)
