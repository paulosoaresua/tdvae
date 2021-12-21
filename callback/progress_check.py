from callback import Callback
from typing import Dict, Set


class ProgressCheck(Callback):

    def __init__(self, display_measures: Set = None, precision: int = 3, update_frequency_type: str = 'batch', update_frequency: int = 100):
        super().__init__(update_frequency_type, update_frequency)
        self._display_measures = display_measures
        self._precision = precision

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        self._print_progress(logs)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        self._print_progress(logs)

    def _print_progress(self, logs: Dict[str, float]):
        progress = f"[Epoch {self._epoch} Batch {self._batch}]"
        if self._display_measures is not None:
            for key, value in logs.items():
                if key in self._display_measures:
                    progress += f" {key}: {value:.{self._precision}f}"
        print(progress)
