from callback import Callback
from callback.logging import Logger
from typing import Dict
from model import TDVAE
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import PIL.Image
from torchvision.transforms import ToTensor


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

    def _on_train_batch_end(self, batch: int, logs: Dict[str, float], train: bool):
        img = self._rollout()
        logs = {'rollout': img}
        self._logger.on_train_batch_end(batch, logs, train)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, float], train: bool):
        img = self._rollout()
        logs = {'rollout': img}
        self._logger.on_train_batch_end(epoch, logs, train)

    def _rollout(self) -> torch.tensor:
        x = next(iter(self._data_loader))[0]
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
        return ToTensor()(image)

