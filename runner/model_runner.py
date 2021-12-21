import torch
import torch.nn  as nn
from model import BaseModel
from typing import List
from callback import Callback
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class ModelRunner:
    def __init__(self, model: BaseModel, optimizer: torch.optim):
        self._model = model
        self._optimizer = optimizer
        self._initial_epoch = 0
        self._initial_batch = 0
        self._random_state_initializers = None

    def reset(self):
        self._initial_epoch = 0
        self._initial_batch = 0
        self._random_state_initializers = None

    def load(self, in_dir: str, save_point: int = None):
        if save_point is None:
            # Retrieve the last saved model in the folder
            pass

        filename = f"{in_dir}/model.{save_point}.pt"
        data_package = torch.load(filename)

        self._model.load_state_dict(data_package['model_state_dict'])
        self._optimizer.load_state_dict(data_package['optimizer_state_dict'])

        # In the first iteration of training, these will be used to preserve the last random states
        # at the time of the model was saved
        # self._random_state_initializers = {
        #     'random_state': data_package['random_state'],
        #     'numpy_random_state': data_package['numpy_random_state'],
        #     'torch_random_state': data_package['torch_random_state'],
        # }
        random.setstate(data_package['random_state'])
        np.random.set_state(data_package['numpy_random_state'])
        torch.set_rng_state(data_package['torch_random_state'])
        self._initial_epoch = data_package['epoch'] + 1

    def train(self, training_set: Dataset, epochs: int, batch_size: int, callbacks: List[Callback]):
        self._model.train()
        self._model.stop_training = False

        training_data_loader = DataLoader(training_set, batch_size=batch_size)

        for callback in callbacks:
            callback.on_training_begin(len(training_data_loader), True)

        logs = {}
        for epoch in range(self._initial_epoch, epochs):
            for callback in callbacks:
                callback.on_train_epoch_begin(epoch, True)

            for batch, (x, y) in enumerate(training_data_loader):
                # Random state initializations if training a loaded model.
                # if self._random_state_initializers is not None:
                #     # In the first iteration over batches in a data loader, an iterable object
                #     # is created and it changes the random state in its constructor.
                #     # That's why we need to keep track of the random state at the end of the previous batch
                #     # and here we need to set the right random state when the model was saved.
                #     random.setstate(self._random_state_initializers['random_state'])
                #     np.random.set_state(self._random_state_initializers['numpy_random_state'])
                #     torch.set_rng_state(self._random_state_initializers['torch_random_state'])
                #     self._random_state_initializers = None

                for callback in callbacks:
                    callback.on_train_batch_begin(batch, True)

                loss = self._model.calculate_loss(x, y)

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._optimizer.step()

                # Other callbacks might overwrite the model internal log if they call the
                # calculate_loss function again (e.g. EarlyStopping)
                logs.update(self._model.log_keys.copy())
                for callback in callbacks:
                    callback.on_train_batch_end(batch, logs, True)

                # logs['last_batch_torch_random_state'] = torch.get_rng_state()

            # self._initial_batch = 0
            # training_data_loader.sampler = IndexedDataLoaderSampler(num_batches, batch_size, 0)

            for callback in callbacks:
                callback.on_train_epoch_end(epoch, logs, True)

            if self._model.stop_training:
                break

        self._model.eval()
