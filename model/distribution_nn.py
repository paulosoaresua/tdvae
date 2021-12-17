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
        self._distribution_params_nns = nn.ModuleList()

        # Value computed in the last call to the forward function
        self._last_computed_params = None

        self._build_h()
        self._build_nn()

    def forward(self, x: torch.tensor) -> torch.tensor:
        h = self._tanh_path(x) * self._sigmoid_path(x)
        # h = self._vanilla_encoder(x)

        distribution_params = []

        for param_idx, linear_layer in enumerate(self._distribution_params_nns):
            # Use the linear layer that branches out of the shared network to compute the values
            # of the posterior parameters
            param_values = linear_layer(h)
            distribution_params.append(param_values)

        self._last_computed_params = tuple(distribution_params)
        return self._last_computed_params

    def sample(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.tensor:
        raise NotImplementedError

    def get_log_likelihood(self, x: torch.tensor, distribution_params: Tuple[torch.tensor, ...]):
        distribution = self._get_distribution(distribution_params)
        return distribution.log_prob(x)

    def _get_distribution(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        raise NotImplementedError

    def _build_h(self):
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

        # vanilla_encoder_modules = [
        #     nn.Linear(self._in_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU()
        # ]
        # self._vanilla_encoder = nn.Sequential(*vanilla_encoder_modules)

    def _build_nn(self):
        raise NotImplementedError
