from torch.utils.tensorboard import SummaryWriter


class LoggerDESGA:
    """
    simple wrapper encapsulating tensorboard logging operations
    """

    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def add_control_performance(self, value, step):
        prefix = "control-performance/"
        self.writer.add_scalar(f"{prefix}return", value, step)

    def add_loss(self, loss, step):
        prefix = "loss/"
        for n, l in loss.items():
            self.writer.add_scalar(f"{prefix}{n}", l, step)

    def add_grad(self, grad_norm, step):
        prefix = "grad-norm/"
        for n, l in grad_norm.items():
            self.writer.add_scalar(f"{prefix}{n}", l, step)

    def add_expected_return(self, perf, step):
        prefix = "performance/"  # name for backward compatibility
        self.writer.add_scalar(f"{prefix}agent", perf, step)

    def add_grad_histograms(self, params_dict, step):
        for net, params in params_dict.items():
            prefix = f"{net}/"
            for name, param in params:
                if param.grad is not None:
                    self.writer.add_histogram(f"{prefix}grad-{name}", param.grad.data, step)
                else:
                    self.writer.add_histogram(f"{prefix}grad-{name}", param.data, step)

    def add_policy_histograms(self, actions, step):
        """
        plots distribution of the parameters of the policy' distribution.
        e.g. in MSD, will plot distribution of logits of the categorical distribution over the actions,
            in MG, will plot the distribution of the parameters of the multi-dimensional gaussian.
        """
        prefix = "policy/"
        for p in range(actions.shape[1]):
            self.writer.add_histogram(f"{prefix}output-{p}", actions[:, p], step)

    def add_system_parameters(self, parameters_dict, step):
        prefix = "systems-params/"
        for name, param in parameters_dict.items():
            self.writer.add_scalar(f"{prefix}{name}", param, step)

    def add_disturbance_parameters(self, parameters_dict, step):
        prefix = "disturbance-params/"
        for name, param in parameters_dict.items():
            self.writer.add_scalar(f"{prefix}{name}", param, step)

    def __del__(self):
        # flush and close the writer before destructing the object
        self.writer.flush()
        self.writer.close()
