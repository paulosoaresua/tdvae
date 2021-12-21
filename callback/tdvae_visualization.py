from callback import Callback
from callback.logging import Logger
from typing import Dict, Any
from model import TDVAE
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
import random


class TDVAEVisualization(Callback):

    def __init__(self, model: TDVAE, logger: Logger, data_set: Dataset, batch_size: int, context_size: int,
                 rollout_size: int):

        # Same frequency as the logger
        super().__init__(logger._update_frequency_type, logger._update_frequency)
        self._model = model
        self._logger = logger
        self._data_loader = DataLoader(data_set, batch_size=batch_size)
        self._batch_size = batch_size
        self._context_size = context_size
        self._rollout_size = rollout_size

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        image = self._rollout()
        self._logger.log_image('rollout', image, self._step, train)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        image = self._rollout()
        self._logger.log_image('rollout', image, self._step, train)

    def _rollout(self) -> torch.tensor:
        # Keep current training random state
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()

        x = next(iter(self._data_loader))[0]
        mode = self._model.training
        self._model.eval()
        with torch.no_grad():
            future = self._model.rollout(x, self._rollout_size)

            fig = plt.figure(0, figsize=(10, 3))
            fig.clf()
            gs = gridspec.GridSpec(self._batch_size, self._context_size + self._rollout_size)
            gs.update(wspace=0.025, hspace=0.025)

            for i in range(self._batch_size):
                # Context
                img = x[i]
                for n in range(self._context_size):
                    axes = plt.subplot(gs[i, n])
                    axes.imshow(img[n, :].reshape(28, 28), cmap='gray')
                    axes.axis('off')

                # Predictions
                img = future[i]
                for n in range(self._rollout_size):
                    axes = plt.subplot(gs[i, n + self._context_size])
                    axes.imshow(img[n, :].reshape(28, 28), cmap='gray')
                    axes.axis('off')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            buffer.seek(0)

            image = PIL.Image.open(buffer)

            self._model.train(mode)

            # Set training random state to its original value
            random.setstate(random_state)
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)

            return ToTensor()(image)

