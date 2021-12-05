import torch.nn as nn
from .exceptions import InvalidActivationFunctionError


def get_activation(activation_name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU()
    }

    try:
        return activations[activation_name]
    except KeyError:
        raise InvalidActivationFunctionError(activation_name)
