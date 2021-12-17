from typing import Any, Dict, Callable
from callback.logging import Logger
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger(Logger):

    def __init__(self, out_dir: str, id: str = None, update_frequency_type: str = 'batch', update_frequency: int = 1,
                 image_transforms: Dict[str, Callable] = None):
        super().__init__(id=id, update_frequency_type=update_frequency_type, update_frequency=update_frequency)
        self._image_transforms = image_transforms

        board_dir = '{}/{}'.format(out_dir, self.id)
        self._writer = SummaryWriter(board_dir)

        self._step = 0

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        self._writer.add_hparams(metric_dict=performance_measures, hparam_dict=hyper_parameters)

    def _on_training_begin(self, train: bool):
        self._step = 0

    def _on_train_batch_end(self, batch: int, logs: Dict[str, float], train: bool):
        self._step += 1
        for key, value in logs.items():
            self._log_measure(key, value, self._step, train)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, float], train: bool):
        self._step += 1
        for key, value in logs.items():
            self._log_measure(key, value, self._step, train)

    def _log_measure(self, measure: str, value: Any, step: int, train: bool):
        measure = f"train/{measure}" if train else f"eval/{measure}"
        if isinstance(value, torch.Tensor):
            image = value
            if self._image_transforms is not None and measure in self._image_transforms:
                image = self._image_transforms[measure](value)
            self._writer.add_image(measure, image, step)
        else:
            self._writer.add_scalar(measure, value, step)

    def __del__(self):
        self._writer.flush()
        self._writer.close()
