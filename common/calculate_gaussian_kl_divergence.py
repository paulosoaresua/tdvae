import torch
from typing import Tuple


def calculate_gaussian_kl_divergence(q_params: Tuple[torch.tensor, ...], p_params: Tuple[torch.tensor, ...]):
    """
    Computes KL(q||p)
    """

    mu_q, log_std_q = q_params
    mu_p, log_std_p = p_params
    mu_diff = mu_q - mu_p
    log_var_diff = 2 * (log_std_q - log_std_p)

    kl = -0.5 * (1.0 + log_var_diff - log_var_diff.exp() - (mu_diff ** 2 / (2 * log_std_p).exp()))
    return torch.sum(kl, dim=1)
