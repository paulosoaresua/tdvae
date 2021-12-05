import torch.nn as nn
import torch
from model import DistributionNN
from typing import Tuple


class GaussianNN(DistributionNN):
    """
    This class contains two single-layer neural nets that encode the parameters of a gaussian distribution.
    """

    def __init__(self, in_features: int, hidden_size: int, latent_size: int):
        super(GaussianNN, self).__init__(in_features, hidden_size)
        self._latent_size = latent_size

        self._build_nn()

    def sample(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        # Sample using the reparametrization trick
        epsilon = torch.rand_like(distribution_params[0])
        samples = distribution_params[0] + torch.exp(distribution_params[1]) * epsilon

        return samples

    def get_distribution(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        return torch.distributions.normal.Normal(distribution_params[0], torch.exp(distribution_params[1]))

    def _build_nn(self):
        # mean and log std branches
        self._distribution_params_nns.append(nn.Linear(self._hidden_size, self._latent_size))
        self._distribution_params_nns.append(nn.Linear(self._hidden_size, self._latent_size))
