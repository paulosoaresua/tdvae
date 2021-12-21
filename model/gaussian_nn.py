import torch.nn as nn
import torch
from model import DistributionNN
from typing import Tuple
import numpy as np


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

    def calculate_log_prob(self, distribution_params: Tuple[torch.tensor, ...], x: torch.tensor):
        mu, log_var = distribution_params
        log_pi = np.log(2.0 * np.pi)
        log_prob = -0.5 * (log_pi + log_var + ((x - mu) ** 2 / log_var.exp()))
        return torch.sum(log_prob, dim=1)

    def calculate_kl_divergence(self, p_params: Tuple[torch.tensor, ...],
                                q_params: Tuple[torch.tensor, ...]) -> torch.distributions:
        mu_p, log_var_p = p_params
        mu_q, log_var_q = q_params

        log_var_diff = log_var_p - log_var_q
        kl = -0.5 * (1.0 + log_var_diff - log_var_diff.exp() - ((mu_p - mu_q) ** 2 / log_var_q.exp()))

        return torch.sum(kl, dim=1)

    def _build_branches(self):
        mu = nn.Linear(self._hidden_size, self._latent_size)
        log_var = nn.Linear(self._hidden_size, self._latent_size)
        self._branch_modules.append(mu)
        self._branch_modules.append(log_var)
