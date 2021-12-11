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

    def sample(self, distribution_params: Tuple[torch.tensor, ...] = None) -> torch.distributions:
        # Sample using the reparametrization trick
        if distribution_params is None:
            distribution_params = self._last_computed_params

        epsilon = torch.rand_like(distribution_params[0])
        samples = distribution_params[0] + torch.exp(distribution_params[1]) * epsilon

        return samples

    def get_kl_divergence(self, other_distribution: 'GaussianNN'):
        mu1, log_sigma1 = self._last_computed_params
        mu2, log_sigma2 = other_distribution._last_computed_params
        kl = (log_sigma2 - log_sigma1) + 0.5 * (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) / torch.exp(
            log_sigma2) ** 2 - 0.5

        return torch.sum(kl, dim=1)

    def _get_distribution(self, distribution_params: Tuple[torch.tensor, ...] = None) -> torch.distributions:
        if distribution_params is None:
            distribution_params = self._last_computed_params

        return torch.distributions.normal.Normal(distribution_params[0], torch.exp(distribution_params[1]))

    def _build_nn(self):
        # mean and log std branches
        self._distribution_params_nns.append(nn.Linear(self._hidden_size, self._latent_size))
        self._distribution_params_nns.append(nn.Linear(self._hidden_size, self._latent_size))
