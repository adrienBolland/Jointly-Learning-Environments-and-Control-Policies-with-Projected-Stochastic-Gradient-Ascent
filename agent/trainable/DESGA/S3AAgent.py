from agent.trainable.DESGA.base import DESGABaseAgent


class S3AAgent(DESGABaseAgent):
    """ Stochastic System Stochastic Action Agent"""

    def __init__(self, InvestmentModel, OperationModel):
        super(S3AAgent, self).__init__(InvestmentModel, OperationModel)

    def get_system(self):
        """Make a decision concerning the system"""
        return self.investment_pol().sample()

    def get_investment_log_prob(self, parameters):
        """Get loglikelihood of an investment decision"""
        return self.investment_pol().log_prob(parameters)

    def get_action(self, observation):
        """Make an operational decision"""
        return self.operation_pol(observation).sample()

    def get_action_log_prob(self, observation, action):
        """Get loglikelihood of an operational decision"""
        return self.operation_pol(observation).log_prob(action)

    def project_parameters(self):
        """Project the parameters on their feasible set"""
        self.investment_pol.project_parameters()
