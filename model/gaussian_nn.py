import torch.nn as nn
import torch
from model import DistributionNN
from typing import Tuple


class GaussianNN(DistributionNN):
    """
    This class contains two single-layer neural nets that encode the parameters of a gaussian distribution.
    """

    def __init__(self, in_features: int, hidden_size: int, latent_size: int):
        self._latent_size = latent_size

        super(GaussianNN, self).__init__(in_features, hidden_size)

    def sample(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        mu, log_var = distribution_params

        epsilon = torch.randn_like(mu)
        samples = mu + torch.exp(0.5 * log_var) * epsilon

        return samples

    def _get_distribution(self, distribution_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        mu, log_var = distribution_params
        return torch.distributions.normal.Normal(mu, torch.exp(0.5 * log_var))

    def _build_nn(self):
        # mean and log std branches
        self._distribution_params_nns.append(nn.Linear(self._hidden_size, self._latent_size))
        self._distribution_params_nns.append(nn.Linear(self._hidden_size, self._latent_size))
