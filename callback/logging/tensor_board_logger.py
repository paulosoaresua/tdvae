from typing import Any, Dict, Callable
from callback.logging import Logger
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger(Logger):

    def __init__(self, out_dir: str, id: str = None, update_frequency_type: str = 'batch', update_frequency: int = 1,
                 image_transforms: Dict[str, Callable] = None):
        super().__init__(id=id, update_frequency_type=update_frequency_type, update_frequency=update_frequency)
        self.image_transforms = image_transforms

        self.board_dir = '{}/{}'.format(out_dir, self.id)
        self._writer = SummaryWriter(self.board_dir)

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        self._writer.add_hparams(metric_dict=performance_measures, hparam_dict=hyper_parameters)

    def log_image(self, measure: str, image: torch.tensor, step: int, train: bool):
        measure = f"train/{measure}" if train else f"eval/{measure}"
        self._writer.add_image(measure, image, step)

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        for key, value in logs.items():
            self._log_measure(key, value, self._step, train)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        for key, value in logs.items():
            self._log_measure(key, value, self._step, train)

    def _log_measure(self, measure: str, value: Any, step: int, train: bool):
        measure = f"train/{measure}" if train else f"eval/{measure}"
        if isinstance(value, float):
            self._writer.add_scalar(measure, value, step)

    def __del__(self):
        self._writer.flush()
        self._writer.close()
