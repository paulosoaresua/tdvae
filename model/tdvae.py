import torch
import torch.nn as nn
from model import GaussianNN, MLP
from typing import List


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

        # Several NNs that comprise a TD_VAE
        self._pre_processing_nn = None

        # Instead of using the built-in stacked LSTMs from torch, I am using two independent models because I
        # need the outputs of both layers to feed into the other modules and torch only outputs the outputs
        # of the last layer in stacked LSTMs
        self._l2_belief_nn = None
        self._l1_belief_nn = None

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        minified_x = self._pre_processing_nn(x)

        l1_beliefs, _ = self._l1_belief_nn(minified_x)
        l2_beliefs, _ = self._l2_belief_nn(l1_beliefs)

        return l1_beliefs, l2_beliefs

    def _build_nn(self):
        self._pre_processing_nn = MLP(self._obs_size, self._minified_obs_size, self._pre_processing_activation,
                                      self._pre_processing_hidden_dims)

        # In the paper they say they use a deep LSTM. Therefore, since we are working with 2 layers for the MNIST
        # problem, I am stacking two LSTMs on top of each other manually to obtain the output of each in the forward
        # method
        self._l1_belief_nn = nn.LSTM(input_size=self._minified_obs_size,
                                     hidden_size=self._belief_size,
                                     batch_first=True)

        # From lower belief to higher belief
        self._l2_belief_nn = nn.LSTM(input_size=self._belief_size,
                                     hidden_size=self._belief_size,
                                     batch_first=True)

        self._l2_belief_to_latent = GaussianNN(self._belief_size, self._d_map_size, self._latent_size)
        # The input to this NN is the belief of the lower layer + the latent state from the higher layer
        self._l1_belief_to_latent = GaussianNN(self._belief_size + self._latent_size, self._d_map_size,
                                               self._latent_size)

        # A latent state at time t1 depends on the beliefs at times t1 and t2 and latent state at time t2
        self._l2_latent_smoothing = GaussianNN(2 * self._belief_size + self._latent_size, self._d_map_size,
                                               self._latent_size)
        # In the lower layer, it also depends on the latent state at time t1 from the higher layer
        self._l1_latent_smoothing = GaussianNN(2 * self._belief_size + 2 * self._latent_size, self._d_map_size,
                                               self._latent_size)

        # From latent state at time t1 to latent state at time t2
        self._l2_latent_transition = GaussianNN(self._latent_size, self._d_map_size, self._latent_size)
        # The lower layer also receives the latent state at time t2 from the higher layer
        self._l1_latent_transition = GaussianNN(2 * self._latent_size, self._d_map_size, self._latent_size)

        # Reconstruct observations at time t2 from the latent state at that time
        self._latent_to_observation = MLP(self._latent_size, self._obs_size, self._decoder_activation,
                                          self._decoder_hidden_dims)

    def _calculate_loss(self, x: torch.tensor, t1: int, t2: int) -> torch.tensor:
        l1_beliefs, l2_beliefs = self(x)

        # beliefs at time t1 and t2
        l2_b_t1 = l2_beliefs[:, t1, :]
        l2_b_t2 = l2_beliefs[:, t2, :]
        l1_b_t1 = l1_beliefs[:, t1, :]
        l1_b_t2 = l1_beliefs[:, t2, :]

        # Sample latent states from beliefs at time t1 and t2
        l2_z_t1 = self._l2_belief_to_latent.sample(self._l2_belief_to_latent(l2_b_t1))
        l2_z_t2 = self._l2_belief_to_latent.sample(self._l2_belief_to_latent(l2_b_t2))

        l1_z_t1 = self._l1_belief_to_latent.sample(self._l2_belief_to_latent(l2_b_t1))
        l1_z_t2 = self._l1_belief_to_latent.sample(self._l2_belief_to_latent(l2_b_t2))

