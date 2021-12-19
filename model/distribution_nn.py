import torch.nn as nn
import torch
from typing import Tuple
from common import NotImplementedError


class DistributionNN(nn.Module):
    """
    In the paper, the authors compute the parameters of desired distributions as f(x) in which x is an embedding
    vector resultant from belief models and smoothing models. f(x) is modeled by g(h(x)), in which
    h = tanh(x)*sigmoid(x) and g is comprised of k linear layers such that each one of them results in one of the parameters of the desired
    distribution (e.g. mean and log std for a Gaussian distribution).

    This class implements h(x) and leaves g(x) to be specified by subclasses representing concrete distributions.
    """

    def __init__(self, in_features: int, hidden_size: int):
        """

        :param in_features: dimension of the embedding vector
        :param hidden_size: number of neurons in the hidden layers of h(x) and g(x)
        :return:
        """
        super(DistributionNN, self).__init__()

        self._in_features = in_features
        self._hidden_size = hidden_size

        self._tanh_path = None
        self._sigmoid_path = None
        self._vanilla_encoder = None

        # To be filled by subclasses
        self._build_torso()

        self._branch_modules = nn.ModuleList()
        self._build_branches()

    def forward(self, x: torch.tensor) -> torch.tensor:
        common_torso = self._tanh_path(x) * self._sigmoid_path(x)

        distribution_params = []
        for branch_layer in self._branch_modules:
            # Use the linear layer that branches out of the shared network to compute the values
            # of the posterior parameters
            param_values = branch_layer(common_torso)
            distribution_params.append(param_values)

        return tuple(distribution_params)

    def sample(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.tensor:
        raise NotImplementedError

    def get_log_likelihood(self, x: torch.tensor, distribution_params: Tuple[torch.tensor, ...]):
        distribution = self._get_distribution(distribution_params)
        return distribution.log_prob(x)

    def _get_distribution(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        raise NotImplementedError

    def _build_torso(self):
        tanh_modules = [
            nn.Linear(self._in_features, self._hidden_size),
            nn.Tanh()
        ]
        self._tanh_path = nn.Sequential(*tanh_modules)

        sigmoid_modules = [
            nn.Linear(self._in_features, self._hidden_size),
            nn.Sigmoid()
        ]
        self._sigmoid_path = nn.Sequential(*sigmoid_modules)

    def _build_branches(self):
        raise NotImplementedError
