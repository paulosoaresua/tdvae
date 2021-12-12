import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GaussianNN, MultilayerLSTM, MLP
from typing import List
from common import Callback, calculate_gaussian_kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class TDVAE(nn.Module):
    def __init__(self, belief_size: int, latent_size: int, d_map_size: int, obs_size: int, minified_obs_size: int,
                 decoder_activation: str, decoder_hidden_dims: List[int], pre_processing_activation: str,
                 pre_processing_hidden_dims: List[int]):
        super(TDVAE, self).__init__()
        self._belief_size = belief_size
        self._latent_size = latent_size
        self._d_map_size = d_map_size
        self._obs_size = obs_size
        self._minified_obs_size = minified_obs_size
        self._decoder_activation = decoder_activation
        self._decoder_hidden_dims = decoder_hidden_dims
        self._pre_processing_activation = pre_processing_activation
        self._pre_processing_hidden_dims = pre_processing_hidden_dims

        self.stop_training = False

        # Several NNs that comprise a TD_VAE
        self._pre_processing_nn = None

        # Custom multilayer LSTM
        self._l2_belief_nn = None

        # Hierarchical model with two layers. Higher layers are sampled first
        # These weights are shared across time but not across layers.
        # They represent P_B distributions.
        self._l2_belief_to_latent = None
        self._l1_belief_to_latent = None

        # Infer latent state at a previous time given the state in the future
        self._l2_latent_smoothing = None
        self._l1_latent_smoothing = None

        # Predict latent state at the future given latent state at the past
        self._l2_latent_transition = None
        self._l1_latent_transition = None

        # MLP that reconstructs observations from latent states
        self._latent_to_observation = None

        self._build_nn()

    def fit(self, training_set: torch.utils.data.dataset, epochs: int, batch_size: int, optimizer: torch.optim,
            callbacks: List[Callback], max_t1: int = 16, max_time_step: int = 4):

        assert max_t1 + max_time_step < training_set.data.size(1)

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

                t1 = np.random.choice(max_t1)
                t2 = np.random.choice(list(range(1, max_time_step + 1))) + t1
                loss, (loss_l1_t1, loss_l1_t2, loss_l2_t1, loss_l2_t2, loss_x), x_t2 = self.calculate_loss(x, t1, t2)
                logs['loss'] = loss.item()
                logs['loss_l1_t1'] = loss_l1_t1.item()
                logs['loss_l1_t2'] = loss_l1_t2.item()
                logs['loss_l2_t1'] = loss_l2_t1.item()
                logs['loss_l2_t2'] = loss_l2_t2.item()
                logs['loss_x'] = loss_x.item()
                logs['x_t2'] = x_t2

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
        minified_x = self._pre_processing_nn(x)
        return self._belief_nn(minified_x)

    def calculate_loss(self, x: torch.tensor, t1: int, t2: int) -> torch.tensor:
        beliefs = self(x)

        # beliefs at time t1 and t2
        l2_b_t1 = beliefs[:, t1, 1, :]
        l2_b_t2 = beliefs[:, t2, 1, :]
        l1_b_t1 = beliefs[:, t1, 0, :]
        l1_b_t2 = beliefs[:, t2, 0, :]

        # Sample latent states from beliefs at time t2
        pb_l2_t2_params = self._l2_belief_to_latent(l2_b_t2)
        l2_z_t2 = self._l2_belief_to_latent.sample(pb_l2_t2_params)

        pb_l1_t2_params = self._l1_belief_to_latent(torch.cat([l1_b_t2, l2_z_t2], dim=1))
        l1_z_t2 = self._l1_belief_to_latent.sample(pb_l1_t2_params)

        z_t2 = torch.cat([l2_z_t2, l1_z_t2], dim=1)

        # Sample latent states at time t1 from samples at time t2
        qs_l2_t1_params = self._l2_latent_smoothing(torch.cat([l2_b_t1, z_t2], dim=1))
        l2_z_t1 = self._l2_latent_smoothing.sample(qs_l2_t1_params)
        qs_l1_t1_params = self._l1_latent_smoothing(torch.cat([l1_b_t1, z_t2, l2_z_t1], dim=1))
        l1_z_t1 = self._l1_latent_smoothing.sample(qs_l1_t1_params)

        z_t1 = torch.cat([l2_z_t1, l1_z_t1], dim=1)

        # Compute parameters of P_B at time t1
        pb_l2_t1_params = self._l2_belief_to_latent(l2_b_t1)
        pb_l1_t1_params = self._l1_belief_to_latent(torch.cat([l1_b_t1, l2_z_t1], dim=1))

        # Compute parameters of P_T at time t2
        pt_l2_t2_params = self._l2_latent_transition(z_t1)
        pt_l1_t2_params = self._l1_latent_transition(torch.cat([z_t1, l2_z_t2], dim=1))

        # Reconstruct observation at time t2 based on state at that time
        x_t2 = torch.sigmoid(self._latent_to_observation(z_t2))

        # Compute terms of the loss
        loss_l1_t1 = calculate_gaussian_kl_divergence(qs_l1_t1_params, pb_l1_t1_params)
        loss_l1_t2 = torch.sum(self._l1_belief_to_latent.get_log_likelihood(
            l1_z_t2, pb_l1_t2_params) - self._l1_latent_transition.get_log_likelihood(l1_z_t2, pt_l1_t2_params), dim=1)

        loss_l2_t1 = calculate_gaussian_kl_divergence(qs_l2_t1_params, pb_l2_t1_params)
        loss_l2_t2 = torch.sum(self._l2_belief_to_latent.get_log_likelihood(
            l2_z_t2, pb_l2_t2_params) - self._l2_latent_transition.get_log_likelihood(l2_z_t2, pt_l2_t2_params), dim=1)

        loss_x = F.binary_cross_entropy(x_t2, x[:, t2, :])

        loss = loss_l1_t1 + loss_l1_t2 + loss_l2_t1 + loss_l2_t2 + loss_x
        loss_terms = (torch.mean(loss_l1_t1), torch.mean(loss_l1_t2), torch.mean(loss_l2_t1),
                      torch.mean(loss_l2_t2), torch.mean(loss_x))

        return torch.mean(loss), loss_terms, x_t2

    def rollout(self, x: torch.tensor, time_steps: int):
        self.eval()
        with torch.no_grad():
            beliefs = self(x)

            # beliefs at time t1 and t2
            t1 = x.size(1) - 1
            l2_b_t1 = beliefs[:, t1, 1, :]
            l1_b_t1 = beliefs[:, t1, 0, :]

            # Sample latent states from beliefs at the last time step of the input
            self._l2_belief_to_latent(l2_b_t1)
            l2_z_t1 = self._l2_belief_to_latent.sample()

            self._l1_belief_to_latent(torch.cat([l1_b_t1, l2_z_t1], dim=1))
            l1_z_t1 = self._l1_belief_to_latent.sample()

            z_t1 = torch.cat([l2_z_t1, l1_z_t1], dim=1)

            # Get predictions for the next time steps
            future_xs = []
            for _ in range(time_steps):
                self._l2_latent_transition(z_t1)
                l2_z_t2_from_t1 = self._l2_latent_transition.sample()
                self._l1_latent_transition(torch.cat([z_t1, l2_z_t2_from_t1], dim=1))
                l1_z_t2_from_t1 = self._l1_latent_transition.sample()

                z_t2 = torch.cat([l2_z_t2_from_t1, l1_z_t2_from_t1], dim=1)

                next_x = torch.sigmoid(self._latent_to_observation(z_t2))
                future_xs.append(next_x)

                z_t1 = z_t2

            future_xs = torch.stack(future_xs, dim=1)

            return future_xs

    def _build_nn(self):
        self._pre_processing_nn = MLP(self._obs_size, self._minified_obs_size, self._pre_processing_activation,
                                      self._pre_processing_hidden_dims, True)

        # In the paper they stack two LSTM on top of each other with transitions between higher layers and lower layers
        self._belief_nn = MultilayerLSTM(self._minified_obs_size, self._belief_size, 2, True)

        self._l2_belief_to_latent = GaussianNN(self._belief_size, self._d_map_size, self._latent_size)
        # The input to this NN is the belief of the lower layer + the latent state from the higher layer
        self._l1_belief_to_latent = GaussianNN(self._belief_size + self._latent_size, self._d_map_size,
                                               self._latent_size)

        # A latent state at time t1 depends on the beliefs at times t1 and t2 and latent state at time t2
        self._l2_latent_smoothing = GaussianNN(self._belief_size + 2 * self._latent_size, self._d_map_size,
                                               self._latent_size)
        # In the lower layer, it also depends on the latent state at time t1 from the higher layer
        self._l1_latent_smoothing = GaussianNN(self._belief_size + 3 * self._latent_size, self._d_map_size,
                                               self._latent_size)

        # From latent state at time t1 to latent state at time t2
        self._l2_latent_transition = GaussianNN(2 * self._latent_size, self._d_map_size, self._latent_size)
        # The lower layer also receives the latent state at time t2 from the higher layer
        self._l1_latent_transition = GaussianNN(3 * self._latent_size, self._d_map_size, self._latent_size)

        # Reconstruct observations at time t2 from the latent state at that time
        self._latent_to_observation = MLP(2 * self._latent_size, self._obs_size, self._decoder_activation,
                                          self._decoder_hidden_dims)


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
