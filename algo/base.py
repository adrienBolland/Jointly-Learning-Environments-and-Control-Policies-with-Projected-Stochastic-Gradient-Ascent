from abc import ABC, abstractmethod


class BaseAlgo(ABC):

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    @abstractmethod
    def initialize(self, **kwargs):
        """ Parameters for performing the optimization """
        raise NotImplementedError

    @abstractmethod
    def fit(self, log_writer=None, device=None):
        """ Execute the optimization """
        raise NotImplementedError
