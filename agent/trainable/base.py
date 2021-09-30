from abc import ABC, abstractmethod

from agent.base import BaseAgent


class TrainableAgent(BaseAgent, ABC):

    def __init__(self):
        super(TrainableAgent, self).__init__()

    @abstractmethod
    def reset_parameters(self):
        """Reset the models parameters"""
        raise NotImplementedError

    @abstractmethod
    def to(self, device):
        """Put the models on a device"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        """Save models"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        """Load models"""
        raise NotImplementedError
