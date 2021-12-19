import numpy as np
import torch

LOG_PI = np.log(2.0 * np.pi)


def calculate_gaussian_log_prob(mu: torch.tensor, log_var: torch.tensor, x: torch.tensor) -> torch.tensor:
    log_prob = -0.5 * (LOG_PI + log_var + ((x - mu) ** 2 / log_var.exp()))
    return torch.sum(log_prob, dim=1)
