import torch
from typing import Tuple


def calculate_gaussian_kl_divergence(p_params: Tuple[torch.tensor, ...], q_params: Tuple[torch.tensor, ...]):
    """
    Computes KL(p||q)

    Different from the torch implementation of the KL divergence, this one works with log vars which I found
    it to me more stable. The gradients of this implementation and the torch one differs in higher decimal places and
    this difference can accumulate over the training phase.
    """

    mu_p, log_var_p = p_params
    mu_q, log_var_q = q_params

    log_var_diff = log_var_p - log_var_q
    kl = -0.5 * (1.0 + log_var_diff - log_var_diff.exp() - ((mu_p - mu_q) ** 2 / log_var_q.exp()))

    return torch.sum(kl, dim=1)
