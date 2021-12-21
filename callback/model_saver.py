from callback import Callback
from typing import Dict, Any
import random
import torch
import numpy as np
from model import BaseModel


class ModelSaver(Callback):

    def __init__(self, model: BaseModel, optimizer: torch.optim, out_dir: str, update_frequency: int = 100):
        # Model saving is fixed per epoch
        super().__init__('epoch', update_frequency)

        self._model = model
        self._optimizer = optimizer
        self._out_dir = out_dir

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        self._save_model()

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        self._save_model()

    def _save_model(self):
        data_package = {'random_state': random.getstate(),
                        'numpy_random_state': np.random.get_state(),
                        'torch_random_state': torch.get_rng_state(),
                        'model_state_dict': self._model.state_dict(),
                        'optimizer_state_dict': self._optimizer.state_dict(),
                        'epoch': self._epoch}

        order = str(int(self._step))
        filename = f"{self._out_dir}/model.{order}.pt"
        torch.save(data_package, filename)
