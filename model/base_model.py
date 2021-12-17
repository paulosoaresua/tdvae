import torch
import torch.nn as nn
from common.exceptions import NotImplementedError


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stop_training = False
        self.log_keys = {}

    def calculate_loss(self, x: torch.tensor, y: torch.tensor):
        raise NotImplementedError
