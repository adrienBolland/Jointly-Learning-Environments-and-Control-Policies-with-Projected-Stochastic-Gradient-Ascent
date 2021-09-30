from agent.trainable.DESGA.base import DESGABaseAgent


class DSSAAgent(DESGABaseAgent):
    """ Deterministic System Stochastic Action Agent"""

    def __init__(self, InvestmentModel, OperationModel):
        super(DSSAAgent, self).__init__(InvestmentModel, OperationModel)

    def get_system(self):
        """Make a decision concerning the system"""
        return self.investment_pol()

    def get_action(self, observation):
        """Make an operational decision"""
        return self.operation_pol(observation).sample()

    def get_action_log_prob(self, observation, action):
        """Get loglikelihood of an operational decision"""
        return self.operation_pol(observation).log_prob(action)

    def get_entropy(self, observation):
        """Get entropy of the distribution"""
        return self.operation_pol(observation).entropy()

    def project_parameters(self):
        """Project the parameters on their feasible set"""
        self.investment_pol.project_parameters()
