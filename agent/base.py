from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self):
        pass

    def __call__(self, observation):
        """Returns an action (forward)"""
        return self.get_action(observation)

    def reset_parameters(self):
        """ reset the models parameters """
        pass

    @abstractmethod
    def get_system(self):
        """Make a decision concerning the system"""
        raise NotImplementedError

    @abstractmethod
    def get_action(self, observation):
        """Make an operational decision"""
        raise NotImplementedError

    @abstractmethod
    def initialize(self, env, **kwargs):
        """Initializes the agent from the environment"""
        raise NotImplementedError

    def save(self, path):
        """Save models"""
        raise NotImplementedError

    def load(self, path):
        """Load models"""
        raise NotImplementedError


