import torch
import torch.nn as nn
from model import BaseModel, GaussianNN, MultilayerLSTM, MLP
from typing import List
from common import calculate_gaussian_kl_divergence, calculate_gaussian_log_prob
from callback import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from pylego import ops


class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TDVAE(BaseModel):
    def __init__(self, belief_size: int, latent_size: int, d_map_size: int, obs_size: int, minified_obs_size: int,
                 decoder_activation: str, decoder_hidden_dims: List[int], pre_processing_activation: str,
                 pre_processing_hidden_dims: List[int], max_time_diff: int, num_layers: int):
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
        self._num_layers = num_layers

        # Several NNs that comprise a TD_VAE
        self._pre_processing_nn = None

        self._belief_network = None

        # Hierarchical model with two layers. Higher layers are sampled first
        # These weights are shared across time but not across layers.
        # They represent P_B distributions.
        self._belief_to_latent_network = None

        # Infer latent state at a previous time given the state in the future
        self._latent_smoothing_network = None

        # Predict latent state at the future given latent state at the past
        self._latent_transition_network = None

        # MLP that reconstructs observations from latent states
        self._latent_to_observation = None

        self._build_nn()

    def forward(self, x: torch.tensor) -> torch.tensor:
        # np.set_printoptions(threshold=np.inf)
        # print(torch.get_rng_state().detach().cpu().numpy())
        # print(torch.randint(low=0, high=20, size=(x.size(0),)))

        # Random time steps with a predefined max range
        t1 = torch.randint(low=0, high=x.size(1) - self._max_time_diff, size=(x.size(0),), device=x.device)
        t2 = t1 + torch.randint(low=1, high=self._max_time_diff + 1, size=(x.size(0),), device=x.device)
        # print(t1)
        # print(t2)

        minified_x = self._pre_processing_nn(x)
        # print(torch.sum(minified_x.clone().detach()))

        beliefs = self._belief_nn(minified_x)
        # print(torch.sum(beliefs.clone().detach()))

        # Extract beliefs at time t1 and t2 (element-wise indexing over the time dimension)
        t1_expanded = t1[..., None, None, None].expand(-1, -1, beliefs.size(2), beliefs.size(3))
        beliefs_t1 = torch.gather(beliefs, 1, t1_expanded).view(-1, beliefs.size(2), beliefs.size(3))
        # print(torch.sum(beliefs_t1.clone().detach()))

        t2_expanded = t2[..., None, None, None].expand(-1, -1, beliefs.size(2), beliefs.size(3))
        beliefs_t2 = torch.gather(beliefs, 1, t2_expanded).view(-1, beliefs.size(2), beliefs.size(3))
        # print(torch.sum(beliefs_t2.clone().detach()))

        # Compute parameters at the distribution pB at time t2 and generate samples from it.
        pb_params_t2 = []
        pb_zs_t2 = []  # per layer
        pb_z_t2 = None
        for layer in range(self._num_layers - 1, -1, -1):
            if layer == self._num_layers - 1:
                params = self._belief_to_latent_network[layer](beliefs_t2[:, layer])
                pb_params_t2 = [[param] for param in params]
            else:
                params = self._belief_to_latent_network[layer](
                    torch.cat([beliefs_t2[:, layer], pb_z_t2], dim=1))
                for i in range(len(params)):
                    pb_params_t2[i].insert(0, params[i])

            pb_z_t2 = self._belief_to_latent_network[layer].sample(params)
            pb_zs_t2.insert(0, pb_z_t2)

        pb_z_t2 = torch.cat(pb_zs_t2, dim=1)
        for i in range(len(pb_params_t2)):
            pb_params_t2[i] = torch.cat(pb_params_t2[i], dim=1)
        pb_params_t2 = tuple(pb_params_t2)
        # print(torch.sum(pb_z_t2.clone().detach()))

        # Compute parameters of the distribution qS at time t1 and generate samples from it
        qs_params_t1 = []
        qs_zs_t1 = []  # per layer
        qs_z_t1 = None
        for layer in range(self._num_layers - 1, -1, -1):
            if layer == self._num_layers - 1:
                params = self._latent_smoothing_network[layer](torch.cat([pb_z_t2, beliefs_t1[:, layer]], dim=1))
                qs_params_t1 = [[param] for param in params]
            else:
                params = self._latent_smoothing_network[layer](
                    torch.cat([pb_z_t2, beliefs_t1[:, layer], qs_z_t1], dim=1))
                for i in range(len(params)):
                    qs_params_t1[i].insert(0, params[i])

            qs_z_t1 = self._latent_smoothing_network[layer].sample(params)
            qs_zs_t1.insert(0, qs_z_t1)

        qs_z_t1 = torch.cat(qs_zs_t1, dim=1)
        for i in range(len(qs_params_t1)):
            qs_params_t1[i] = torch.cat(qs_params_t1[i], dim=1)
        qs_params_t1 = tuple(qs_params_t1)
        # print(torch.sum(qs_z_t1.clone().detach()))

        # Compute parameters of the distribution pB at time t1
        pb_params_t1 = []
        for layer in range(self._num_layers - 1, -1, -1):
            if layer == self._num_layers - 1:
                params = self._belief_to_latent_network[layer](beliefs_t1[:, layer])
                pb_params_t1 = [[param] for param in params]
            else:
                params = self._belief_to_latent_network[layer](
                    torch.cat([beliefs_t1[:, layer], qs_zs_t1[layer + 1]], dim=1))
                for i in range(len(params)):
                    pb_params_t1[i].insert(0, params[i])

        for i in range(len(pb_params_t1)):
            pb_params_t1[i] = torch.cat(pb_params_t1[i], dim=1)
        pb_params_t1 = tuple(pb_params_t1)

        # Compute parameters of the distribution pT at time t2
        pt_params_t2 = []
        for layer in range(self._num_layers - 1, -1, -1):
            if layer == self._num_layers - 1:
                params = self._latent_transition_network[layer](qs_z_t1)
                pt_params_t2 = [[param] for param in params]
            else:
                params = self._latent_transition_network[layer](
                    torch.cat([qs_z_t1, pb_zs_t2[layer + 1]], dim=1))
                for i in range(len(params)):
                    pt_params_t2[i].insert(0, params[i])

        for i in range(len(pt_params_t2)):
            pt_params_t2[i] = torch.cat(pt_params_t2[i], dim=1)
        pt_params_t2 = tuple(pt_params_t2)

        # Reconstruct observation at time t2 based on belief state (sample) at time t2
        x_t2 = torch.sigmoid(self._latent_to_observation(pb_z_t2))

        return pb_params_t1, qs_params_t1, pb_params_t2, pt_params_t2, pb_z_t2, x_t2, t2

    def calculate_loss(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        pb_params_t1, qs_params_t1, pb_params_t2, pt_params_t2, pb_z_t2, x_t2, t2 = self(x)

        # Functions for Gaussian distributions in the original formulation
        kl_divergence = self._belief_to_latent_network[0].calculate_kl_divergence
        log_prob = self._belief_to_latent_network[0].calculate_log_prob

        # KL(qs, pb)
        kl_smoothing_loss = kl_divergence(qs_params_t1, pb_params_t1).mean()

        # log_pb(z_t2) - log_pt(z_t2)
        kl_prediction_loss = (log_prob(pb_params_t2, pb_z_t2) - log_prob(pt_params_t2, pb_z_t2)).mean()

        kl_loss = kl_smoothing_loss + kl_prediction_loss

        # Reconstruction loss
        t2_expanded = t2[..., None, None].expand(-1, -1, x.size(2))
        true_x_t2 = torch.gather(x, 1, t2_expanded).view(-1, x.size(2))

        bce_loss = nn.BCELoss(reduction='sum')(x_t2, true_x_t2) / x.size(0)
        bce_optimal_loss = nn.BCELoss(reduction='sum')(true_x_t2, true_x_t2).detach() / x.size(0)
        reconstruction_loss = bce_loss - bce_optimal_loss

        # Final TD-VAE loss
        total_loss = kl_loss + reconstruction_loss

        # Store current computation for log purposes
        self.log_keys['kl_smoothing_loss'] = kl_smoothing_loss.item()
        self.log_keys['kl_prediction_loss'] = kl_prediction_loss.item()
        self.log_keys['kl_loss'] = kl_loss.item()
        self.log_keys['bce_loss'] = bce_loss.item()
        self.log_keys['bce_optimal_loss'] = bce_optimal_loss.item()
        self.log_keys['reconstruction_loss'] = reconstruction_loss.item()
        self.log_keys['total_loss'] = total_loss.item()

        return total_loss

    def rollout(self, x: torch.tensor, time_steps: int):
        tmp = self._pre_processing_nn(x)
        beliefs = self._belief_nn(tmp)

        # Last time step as initial point
        t = x.size(1) - 1

        # Sample latent states from beliefs at the last time step of the input
        zs = []
        z = None
        for layer in range(self._num_layers - 1, -1, -1):
            if layer == self._num_layers - 1:
                params = self._belief_to_latent_network[layer](beliefs[:, t, layer])
            else:
                params = self._belief_to_latent_network[layer](torch.cat([beliefs[:, t, layer], z], dim=1))

            z = self._belief_to_latent_network[layer].sample(params)
            zs.insert(0, z)

        z_start = torch.cat(zs, dim=1)

        # Get predictions for the next time steps
        future_xs = []
        for _ in range(time_steps):
            zs = []
            z = None
            for layer in range(self._num_layers - 1, -1, -1):
                if layer == self._num_layers - 1:
                    params = self._latent_transition_network[layer](z_start)
                else:
                    params = self._latent_transition_network[layer](torch.cat([z_start, z], dim=1))

                z = self._latent_transition_network[layer].sample(params)
                zs.insert(0, z)

            z = torch.cat(zs, dim=1)

            next_x = torch.sigmoid(self._latent_to_observation(z))
            future_xs.append(next_x)

            z_start = z

        future_xs = torch.stack(future_xs, dim=1)

        return future_xs

    def _build_nn(self):
        self._pre_processing_nn = MLP(self._obs_size, self._minified_obs_size, self._pre_processing_activation,
                                      self._pre_processing_hidden_dims, True)

        # Stacked LSTMs with across-time connections between higher and lower layers.
        self._belief_nn = MultilayerLSTM(self._minified_obs_size, self._belief_size, 2, True, True)

        # Extracting belief states from a belief
        # The input to this NN is the belief of the lower layer + the latent state from the higher layer
        self._belief_to_latent_network = nn.ModuleList()
        for layer in range(self._num_layers):
            # Lower layers receive the output of higher layers as input as well
            input_size = self._belief_size + self._latent_size * (self._num_layers - layer - 1)
            self._belief_to_latent_network.append(GaussianNN(input_size, self._d_map_size, self._latent_size))

        # A latent state at time t1 depends on the beliefs at times t1 and t2 and latent state at time t2
        # In the lower layer, it also depends on the latent state at time t1 from the higher layer
        self._latent_smoothing_network = nn.ModuleList()
        for layer in range(self._num_layers):
            input_size = self._belief_size + 2 * self._latent_size + self._latent_size * (self._num_layers - layer - 1)
            self._latent_smoothing_network.append(GaussianNN(input_size, self._d_map_size, self._latent_size))

        # From latent state at time t1 to latent state at time t2
        # The lower layer also receives the latent state at time t2 from the higher layer
        self._latent_transition_network = nn.ModuleList()
        for layer in range(self._num_layers):
            input_size = 2 * self._latent_size + self._latent_size * (self._num_layers - layer - 1)
            self._latent_transition_network.append(GaussianNN(input_size, self._d_map_size, self._latent_size))

        # Reconstruct observations at time t2 from the latent state at that time
        self._latent_to_observation = MLP(2 * self._latent_size, self._obs_size, self._decoder_activation,
                                          self._decoder_hidden_dims)
