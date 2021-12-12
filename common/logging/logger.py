from typing import Dict
import datetime
from common import Callback


class Logger(Callback):
    def __init__(self,
                 id: str = None,
                 update_frequency_type: str = 'batch',
                 update_frequency: int = 100):
        super(Logger, self).__init__()
        if id is None:
            self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.id = id

        self._update_frequency_type = update_frequency_type
        self._update_frequency = update_frequency

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        raise NotImplementedError


