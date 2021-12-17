import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GaussianNN, MultilayerLSTM, MLP
from typing import List
from common import calculate_gaussian_kl_divergence
from callback import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class VAE(nn.Module):
    def __init__(self, latent_size: int, d_map_size: int, obs_size: int, decoder_activation: str,
                 decoder_hidden_dims: List[int]):
        super(VAE, self).__init__()
        self._latent_size = latent_size
        self._d_map_size = d_map_size
        self._obs_size = obs_size
        self._decoder_activation = decoder_activation
        self._decoder_hidden_dims = decoder_hidden_dims

        self.stop_training = False

        self._encoder = None
        self._decoder = None

        self._build_nn()

    def fit(self, training_set: torch.utils.data.dataset, epochs: int, batch_size: int, optimizer: torch.optim,
            callbacks: List[Callback], beta: float = 1):

        training_data_loader = DataLoader(training_set, batch_size=batch_size)

        self.train()
        self.stop_training = False

        for callback in callbacks:
            callback.on_training_begin(self)

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_train_epoch_begin(epoch)

            pbar_data = tqdm(training_data_loader, ascii=True,
                             desc="[{current_epochs:0{epoch_length}d}/{total_epochs:0{epoch_length}}]".format(
                                 epoch_length=len(str(epochs)),
                                 current_epochs=epoch + 1,
                                 total_epochs=epochs))

            logs = {}
            for batch, (x, _) in enumerate(pbar_data):
                for callback in callbacks:
                    callback.on_train_batch_begin(batch)

                loss, kl, loss_x, x_hat = self.calculate_loss(x, beta)
                logs['loss'] = loss.item()
                logs['kl'] = kl.item()
                logs['loss_x'] = loss_x.item()
                logs['x_hat'] = x_hat

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for callback in callbacks:
                    callback.on_train_batch_end(batch, logs)

                # Update progress bar
                postfix = {}
                postfix["loss"] = logs['loss']
                pbar_data.set_postfix(postfix, refresh=False)

            for callback in callbacks:
                callback.on_train_epoch_end(epoch, logs)

            if self.stop_training:
                break

        self.eval()

    def forward(self, x: torch.tensor) -> torch.tensor:
        params = self._encoder(x)
        z = self._encoder.sample(params)
        x_hat = torch.sigmoid(self._decoder(z))

        return params, z, x_hat

    def calculate_loss(self, x: torch.tensor, beta: float) -> torch.tensor:
        params, z, x_hat = self(x)

        # KL(q||N(0,1))
        kl = calculate_gaussian_kl_divergence(params, [torch.zeros_like(z), torch.zeros_like(z)]).mean()

        loss_x = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(1)

        loss = beta * kl + loss_x

        return loss, kl, loss_x, x_hat

    def generate_image(self, num_samples: int):
        self.eval()
        with torch.no_grad():
            z = torch.rand_like(torch.zeros([num_samples, self._latent_size]))
            return self._decoder(z)

    def _build_nn(self):
        self._encoder = GaussianNN(self._obs_size, self._d_map_size, self._latent_size)
        self._decoder = MLP(self._latent_size, self._obs_size, self._decoder_activation, self._decoder_hidden_dims)


if __name__ == '__main__':
    import torchvision
    from common.transformation import RolloutImage
    from torch.utils.data import DataLoader

    training_set = torchvision.datasets.MNIST('data/',
                                              train=True,
                                              download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  RolloutImage(time_steps=20),
                                                  torchvision.transforms.Lambda(
                                                      lambda x: torch.flatten(x, start_dim=1))
                                              ]))

    training_data_loader = DataLoader(training_set, batch_size=3)

    for x, y in training_data_loader:
        print(x.shape)
