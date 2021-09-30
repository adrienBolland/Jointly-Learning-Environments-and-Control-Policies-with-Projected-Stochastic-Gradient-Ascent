import torch


class ClipGrad(torch.autograd.Function):

    MAX_NORM = 10.0e+10

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        norm = torch.linalg.norm(grad_output, 2)
        return grad_output / (norm.clamp(min=ClipGrad.MAX_NORM) - ClipGrad.MAX_NORM + 1)


def gradient_norm(parameters):
    total_norm = 0.
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm
