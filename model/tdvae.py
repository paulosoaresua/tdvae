import torch
import torch.nn as nn
from model import BaseModel, GaussianNN, MultilayerLSTM, MLP
from typing import List
from common import calculate_gaussian_kl_divergence
from callback import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F



class TDVAE(BaseModel):
    def __init__(self, belief_size: int, latent_size: int, d_map_size: int, obs_size: int, minified_obs_size: int,
                 decoder_activation: str, decoder_hidden_dims: List[int], pre_processing_activation: str,
                 pre_processing_hidden_dims: List[int], max_time_diff: int):
        super().__init__()
        self._belief_size = belief_size
        self._latent_size = latent_size
        self._d_map_size = d_map_size
        self._obs_size = obs_size
        self._minified_obs_size = minified_obs_size
        self._decoder_activation = decoder_activation
        self._decoder_hidden_dims = decoder_hidden_dims
        self._pre_processing_activation = pre_processing_activation
        self._pre_processing_hidden_dims = pre_processing_hidden_dims
        self._max_time_diff = max_time_diff

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        minified_x = self._pre_processing_nn(x)
        return self._belief_nn(minified_x)

    def calculate_loss(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:

        # Random time steps with a predefined max range
        t1 = torch.randint(low=0, high=x.size(1) - self._max_time_diff, size=(x.size(0),), device=x.device)
        t2 = t1 + torch.randint(low=1, high=self._max_time_diff + 1, size=(x.size(0),), device=x.device)

        # Calculate beliefs
        beliefs = self(x)

        # Extract beliefs at time t1 and t2 (element-wise indexing over the time dimension)
        t1_expanded = t1[..., None, None, None].expand(-1, -1, beliefs.size(2), beliefs.size(3))
        beliefs_t1 = torch.gather(beliefs, 1, t1_expanded).view(-1, beliefs.size(2), beliefs.size(3))

        t2_expanded = t2[..., None, None, None].expand(-1, -1, beliefs.size(2), beliefs.size(3))
        beliefs_t2 = torch.gather(beliefs, 1, t2_expanded).view(-1, beliefs.size(2), beliefs.size(3))

        l2_b_t1 = beliefs_t1[:, 1]
        l2_b_t2 = beliefs_t2[:, 1]
        l1_b_t1 = beliefs_t1[:, 0]
        l1_b_t2 = beliefs_t2[:, 0]

        # Sample latent states from beliefs at time t2
        pb_l2_t2_params = self._l2_belief_to_latent(l2_b_t2)
        l2_z_t2 = self._l2_belief_to_latent.sample(pb_l2_t2_params)

        pb_l1_t2_params = self._l1_belief_to_latent(torch.cat([l1_b_t2, l2_z_t2], dim=1))
        l1_z_t2 = self._l1_belief_to_latent.sample(pb_l1_t2_params)

        # Aggregate samples at time t2
        z_t2 = torch.cat([l1_z_t2, l2_z_t2], dim=1)

        # Sample latent states at time t1 from samples at time t2
        qs_l2_t1_params = self._l2_latent_smoothing(torch.cat([z_t2, l2_b_t1], dim=1))
        l2_z_t1 = self._l2_latent_smoothing.sample(qs_l2_t1_params)

        qs_l1_t1_params = self._l1_latent_smoothing(torch.cat([z_t2, l1_b_t1, l2_z_t1], dim=1))
        l1_z_t1 = self._l1_latent_smoothing.sample(qs_l1_t1_params)

        # Aggregate samples at time t1
        z_t1 = torch.cat([l1_z_t1, l2_z_t1], dim=1)

        # Compute parameters of P_B at time t1
        pb_l2_t1_params = self._l2_belief_to_latent(l2_b_t1)
        pb_l1_t1_params = self._l1_belief_to_latent(torch.cat([l1_b_t1, l2_z_t1], dim=1))

        # Compute parameters of P_T at time t2
        pt_l2_t2_params = self._l2_latent_transition(z_t1)
        pt_l1_t2_params = self._l1_latent_transition(torch.cat([z_t1, l2_z_t2], dim=1))

        # Reconstruct observation at time t2 based on state at that time
        x_t2 = self._latent_to_observation(z_t2)

        # Compute terms of the loss

        # KL(qs, pb)
        loss_l2_t1 = calculate_gaussian_kl_divergence(qs_l2_t1_params, pb_l2_t1_params).mean()
        loss_l1_t1 = calculate_gaussian_kl_divergence(qs_l1_t1_params, pb_l1_t1_params).mean()

        # log(pb(t2)) - log(pt(t2))
        loss_l2_t2 = torch.sum(self._l2_belief_to_latent.get_log_likelihood(
            l2_z_t2, pb_l2_t2_params) - self._l2_latent_transition.get_log_likelihood(l2_z_t2, pt_l2_t2_params),
                               dim=1).mean()
        loss_l1_t2 = torch.sum(self._l1_belief_to_latent.get_log_likelihood(
            l1_z_t2, pb_l1_t2_params) - self._l1_latent_transition.get_log_likelihood(l1_z_t2, pt_l1_t2_params),
                               dim=1).mean()

        # KL terms
        kl_t1_loss = loss_l1_t1 + loss_l2_t1
        kl_t2_loss = loss_l1_t2 + loss_l2_t2
        kl_loss = kl_t1_loss + kl_t2_loss

        # Reconstruction loss
        t2_expanded = t2[..., None, None].expand(-1, -1, x.size(2))
        true_x_t2 = torch.gather(x, 1, t2_expanded).view(-1, x.size(2))

        bce_loss = nn.BCEWithLogitsLoss(reduction='sum')(x_t2, true_x_t2) / x.size(0)
        bce_optimal_loss = nn.BCELoss(reduction='sum')(true_x_t2, true_x_t2).detach() / x.size(0)
        x_loss = bce_loss - bce_optimal_loss

        total_loss = kl_loss + x_loss

        # Store current computation for log purposes
        self.log_keys['loss_l1_t1'] = loss_l1_t1.item()
        self.log_keys['loss_l2_t1'] = loss_l2_t1.item()
        self.log_keys['loss_l1_t2'] = loss_l1_t2.item()
        self.log_keys['loss_l2_t2'] = loss_l2_t2.item()
        self.log_keys['kl_t1_loss'] = kl_t1_loss.item()
        self.log_keys['kl_t2_loss'] = kl_t2_loss.item()
        self.log_keys['kl_loss'] = kl_loss.item()
        self.log_keys['bce_loss'] = bce_loss.item()
        self.log_keys['bce_optimal_loss'] = bce_optimal_loss.item()
        self.log_keys['x_loss'] = x_loss.item()
        self.log_keys['total_loss'] = total_loss.item()

        return total_loss

    def rollout(self, x: torch.tensor, time_steps: int):
        self.eval()
        with torch.no_grad():
            beliefs = self(x)

            # beliefs at time t1 and t2
            t1 = x.size(1) - 1
            l2_b_t1 = beliefs[:, t1, 1, :]
            l1_b_t1 = beliefs[:, t1, 0, :]

            # Sample latent states from beliefs at the last time step of the input
            l2_b_t1_params = self._l2_belief_to_latent(l2_b_t1)
            l2_z_t1 = self._l2_belief_to_latent.sample(l2_b_t1_params)

            l1_b_t1_params = self._l1_belief_to_latent(torch.cat([l1_b_t1, l2_z_t1], dim=1))
            l1_z_t1 = self._l1_belief_to_latent.sample(l1_b_t1_params)

            z_t1 = torch.cat([l1_z_t1, l2_z_t1], dim=1)

            # Get predictions for the next time steps
            future_xs = []
            for _ in range(time_steps):
                pt_l2_params = self._l2_latent_transition(z_t1)
                l2_z_t2_from_t1 = self._l2_latent_transition.sample(pt_l2_params)
                pt_l1_params = self._l1_latent_transition(torch.cat([z_t1, l2_z_t2_from_t1], dim=1))
                l1_z_t2_from_t1 = self._l1_latent_transition.sample(pt_l1_params)

                z_t2 = torch.cat([l1_z_t2_from_t1, l2_z_t2_from_t1], dim=1)

                next_x = torch.sigmoid(self._latent_to_observation(z_t2))
                future_xs.append(next_x)

                z_t1 = z_t2

            future_xs = torch.stack(future_xs, dim=1)

            return future_xs

    def _build_nn(self):
        self._pre_processing_nn = MLP(self._obs_size, self._minified_obs_size, self._pre_processing_activation,
                                      self._pre_processing_hidden_dims, True)

        # In the paper they stack two LSTM on top of each other with transitions between higher layers and lower layers
        self._belief_nn = MultilayerLSTM(self._minified_obs_size, self._belief_size, 2, True, True)

        # The input to this NN is the belief of the lower layer + the latent state from the higher layer
        self._l1_belief_to_latent = GaussianNN(self._belief_size + self._latent_size, self._d_map_size,
                                               self._latent_size)
        self._l2_belief_to_latent = GaussianNN(self._belief_size, self._d_map_size, self._latent_size)

        # A latent state at time t1 depends on the beliefs at times t1 and t2 and latent state at time t2
        # In the lower layer, it also depends on the latent state at time t1 from the higher layer
        self._l1_latent_smoothing = GaussianNN(self._belief_size + 3 * self._latent_size, self._d_map_size,
                                               self._latent_size)
        self._l2_latent_smoothing = GaussianNN(self._belief_size + 2 * self._latent_size, self._d_map_size,
                                               self._latent_size)

        # From latent state at time t1 to latent state at time t2
        # The lower layer also receives the latent state at time t2 from the higher layer
        self._l1_latent_transition = GaussianNN(3 * self._latent_size, self._d_map_size, self._latent_size)
        self._l2_latent_transition = GaussianNN(2 * self._latent_size, self._d_map_size, self._latent_size)

        # Reconstruct observations at time t2 from the latent state at that time
        self._latent_to_observation = MLP(2 * self._latent_size, self._obs_size, self._decoder_activation,
                                          self._decoder_hidden_dims)
