from typing import Dict
import datetime
from callback import Callback


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

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        raise NotImplementedError


