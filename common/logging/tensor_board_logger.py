from typing import Any, Dict, Callable
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger(Logger):

    def __init__(self, out_dir: str, image_transforms: Dict[str, Callable], *args, **kwargs):
        super(TensorBoardLogger, self).__init__(*args, **kwargs)
        self._image_transforms = image_transforms

        board_dir = '{}/{}'.format(out_dir, self.id)
        self._writer = SummaryWriter(board_dir)

        self._step = 0

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        self._writer.add_hparams(metric_dict=performance_measures, hparam_dict=hyper_parameters)

    def on_training_begin(self, model: Any):
        self._step = 0

    def on_train_batch_end(self, batch: int, logs: Dict[str, float]):
        if self._update_frequency_type == 'batch' and batch % self._update_frequency == 0:
            self._step += 1
            for key, value in logs.items():
                self._log_measure(key, value, self._step)

    def on_train_epoch_end(self, epoch: int, logs: Dict[str, float]):
        if self._update_frequency_type == 'epoch' and epoch % self._update_frequency == 0:
            self._step += 1
            for key, value in logs.items():
                self._log_measure(key, value, self._step)

    def __del__(self):
        self._writer.flush()
        self._writer.close()

    def _log_measure(self, measure: str, value: Any, step: int):
        if isinstance(value, torch.Tensor):
            if measure in self._image_transforms:
                tensor_image = self._image_transforms[measure](value)
                self._writer.add_image(measure, tensor_image, step)
        else:
            self._writer.add_scalar(measure, value, step)
