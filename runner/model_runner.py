import torch
import torch.nn  as nn
from model import BaseModel
from typing import List
from callback import Callback
from torch.utils.data import Dataset, DataLoader


class ModelRunner:
    def __init__(self, model: BaseModel):
        self._model = model

    def train(self, training_set: Dataset, epochs: int, batch_size: int, optimizer: torch.optim,
              callbacks: List[Callback]):

        self._model.train()
        self._model.stop_training = False

        training_data_loader = DataLoader(training_set, batch_size=batch_size)

        for callback in callbacks:
            callback.on_training_begin(True)

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_train_epoch_begin(epoch, True)

            logs = {}
            for batch, (x, y) in enumerate(training_data_loader):

                for callback in callbacks:
                    callback.on_train_batch_begin(batch, True)

                loss = self._model.calculate_loss(x, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optimizer.step()

                # Other callbacks might overwrite the model internal log if they call the
                # calculate_loss function again (e.g. EarlyStopping)
                logs = self._model.log_keys.copy()
                for callback in callbacks:
                    callback.on_train_batch_end(batch, logs, True)

                # if batch == 1:
                #     return

            for callback in callbacks:
                callback.on_train_epoch_end(epoch, logs, True)

            if self._model.stop_training:
                break

        self._model.eval()

