from typing import Dict, Any
import datetime
from callback import Callback
import torch


class Logger(Callback):
    def __init__(self,
                 id: str = None,
                 update_frequency_type: str = 'batch',
                 update_frequency: int = 1):
        super().__init__(update_frequency_type, update_frequency)
        if id is None:
            self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.id = id

    def log_hyper_parameters(self, performance_measures: Dict[str, Any], hyper_parameters: Dict[str, float]):
        raise NotImplementedError

    def log_image(self, measure: str, value: torch.tensor, step: int, train: bool):
        raise NotImplementedError



