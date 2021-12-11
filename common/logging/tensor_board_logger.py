from typing import Dict
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
import os
from model import TDVAE


class TensorBoardLogger(Logger):

    def __init__(self, out_dir: str, *args, **kwargs):
        super(TensorBoardLogger, self).__init__(*args, **kwargs)
        board_dir = '{}/{}'.format(out_dir, self._id)
        self._writer = SummaryWriter(board_dir)

        self._images_dir = '{}/images'.format(board_dir)
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        self._writer.add_hparams(metric_dict=performance_measures, hparam_dict=hyper_parameters)

    def on_train_batch_end(self, batch: int, logs: Dict[str, float]):
        if self._update_frequency_type == 'batch' and batch % self._update_frequency == 0:
            for key, value in logs.items():
                self._writer.add_scalar(key, value, batch)

    def on_train_epoch_end(self, epoch: int, logs: Dict[str, float]):
        if self._update_frequency_type == 'epoch' and epoch % self._update_frequency == 0:
            for key, value in logs.items():
                self._writer.add_scalar(key, value, epoch)

    def __del__(self):
        self._writer.flush()
        self._writer.close()
