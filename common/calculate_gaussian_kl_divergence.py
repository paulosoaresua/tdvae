import torch
from typing import Tuple


def calculate_gaussian_kl_divergence(p_params: Tuple[torch.tensor, ...], q_params: Tuple[torch.tensor, ...]):
    """
    Computes KL(p||q)
    """

    mu_p, log_var_p = p_params
    mu_q, log_var_q = q_params

    p = torch.distributions.normal.Normal(mu_p, torch.exp(0.5 * log_var_p))
    q = torch.distributions.normal.Normal(mu_q, torch.exp(0.5 * log_var_q))

    kl = torch.distributions.kl.kl_divergence(p, q)
    return torch.sum(kl, dim=1)
