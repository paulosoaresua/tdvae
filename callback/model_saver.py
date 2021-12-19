from callback import Callback
from typing import Dict, Set


class ProgressCheck(Callback):

    def __init__(self, out_dir: str, update_frequency_type: str = 'batch', update_frequency: int = 100):
        super().__init__(update_frequency_type, update_frequency)

    def _on_train_batch_end(self, batch: int, logs: Dict[str, float], train: bool):
        self._print_progress(logs)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, float], train: bool):
        self._print_progress(logs)

    def _print_progress(self, logs: Dict[str, float]):
        progress = f"[Epoch {self._epoch} Batch {self._batch}]"
        if self._display_measures is not None:
            for key, value in logs.items():
                if key in self._display_measures:
                    progress += f" {key}: {value:.4f}"
        print(progress)
