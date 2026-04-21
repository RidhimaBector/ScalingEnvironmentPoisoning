"""
Variational Autoencoder (VAE) for encoding victim behavior traces.

Based on "Poisoning the Well: Can We Simultaneously Attack a Group of Learning Agents?" (ALA 2023)

The VAE encodes a behavior trace (nS x 2 matrix of [state, last_action]) into a latent
Gaussian distribution (mu, log_var). This enables size-agnostic aggregation of
multiple victims in collective attack settings.
"""

import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE_Encoder(nn.Module):
    """Encodes behavior trace (nS x 2) -> (mean, log_var) of latent Gaussian.

    Args:
        input_size: Size of flattened behavior trace (nS * 2)
        hidden_size: Size of hidden layers
        latent_size: Dimension of latent space
    """

    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super(VAE_Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)

        # Output mean and log_var
        self.fc_mu = nn.Linear(hidden_size // 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size // 2, latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns (mu, log_var).

        Args:
            x: Input tensor of shape (batch, input_size) or (batch, nS, 2)

        Returns:
            Tuple of (mu, log_var), each of shape (batch, latent_size)
        """
        # Flatten if needed
        x = x.view(-1, self.input_size)

        # Encode
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))

        # Get distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)

        return mu, log_var


class VAE_Decoder(nn.Module):
    """Decodes latent sample -> action probabilities per state.

    Args:
        latent_size: Dimension of latent space
        hidden_size: Size of hidden layers
        output_size: Size of output (nS * nA for action probabilities)
        n_states: Number of states
        n_actions: Number of actions
    """

    def __init__(self, latent_size: int, hidden_size: int, n_states: int, n_actions: int):
        super(VAE_Decoder, self).__init__()
        self.latent_size = latent_size
        self.n_states = n_states
        self.n_actions = n_actions
        output_size = n_states * n_actions

        # Decoder network
        self.fc1 = nn.Linear(latent_size, hidden_size // 2)
        self.ln1 = nn.LayerNorm(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to action probabilities.

        Args:
            z: Latent sample of shape (batch, latent_size)

        Returns:
            Action probabilities of shape (batch, n_states, n_actions)
        """
        h = F.relu(self.ln1(self.fc1(z)))
        h = F.relu(self.ln2(self.fc2(h)))
        logits = self.fc3(h)

        # Reshape to (batch, n_states, n_actions) and apply softmax per state
        logits = logits.view(-1, self.n_states, self.n_actions)
        probs = F.softmax(logits, dim=2)

        return probs


class BehaviorVAE:
    """Complete VAE wrapper for behavior encoding with train, encode, and save/load.

    Args:
        n_states: Number of states in the environment
        n_actions: Number of actions in the environment
        hidden_size: Size of hidden layers
        latent_size: Dimension of latent space
        learning_rate: Learning rate for optimizer
        beta: KL divergence weight (beta-VAE)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden_size: int = 128,
        latent_size: int = 8,
        learning_rate: float = 0.001,
        beta: float = 1.0
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.beta = beta

        # Input size: state index + action (2 values per state)
        self.input_size = n_states * 2

        # Initialize encoder and decoder
        self.encoder = VAE_Encoder(self.input_size, hidden_size, latent_size).to(device)
        self.decoder = VAE_Decoder(latent_size, hidden_size, n_states, n_actions).to(device)

        # Combined parameters for optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through VAE.

        Returns:
            Tuple of (reconstructed, mu, log_var)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def compute_loss(
        self,
        behavior_trace: torch.Tensor,
        target_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE loss = reconstruction + KL divergence.

        Args:
            behavior_trace: Input behavior trace (batch, nS, 2)
            target_actions: Target action indices (batch, nS)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        recon_probs, mu, log_var = self.forward(behavior_trace)

        # Reconstruction loss: cross-entropy for action prediction
        # recon_probs: (batch, n_states, n_actions)
        # target_actions: (batch, n_states)
        recon_loss = F.cross_entropy(
            recon_probs.view(-1, self.n_actions),
            target_actions.view(-1).long(),
            reduction='mean'
        )

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        }

        return total_loss, loss_dict

    def train_step(
        self,
        behavior_traces: np.ndarray,
        target_actions: np.ndarray
    ) -> Dict[str, float]:
        """Single training step on a batch.

        Args:
            behavior_traces: Batch of behavior traces (batch, nS, 2)
            target_actions: Batch of target actions (batch, nS)

        Returns:
            Dictionary of loss values
        """
        self.encoder.train()
        self.decoder.train()

        # Convert to tensors
        traces_tensor = torch.FloatTensor(behavior_traces).to(device)
        actions_tensor = torch.LongTensor(target_actions).to(device)

        # Compute loss and backprop
        self.optimizer.zero_grad()
        loss, loss_dict = self.compute_loss(traces_tensor, actions_tensor)
        loss.backward()
        self.optimizer.step()

        return loss_dict

    def train_epoch(
        self,
        behavior_traces: np.ndarray,
        target_actions: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Train for one epoch on the dataset.

        Args:
            behavior_traces: All behavior traces (N, nS, 2)
            target_actions: All target actions (N, nS)
            batch_size: Batch size for training

        Returns:
            Average loss values for the epoch
        """
        # Create dataloader
        traces_tensor = torch.FloatTensor(behavior_traces)
        actions_tensor = torch.LongTensor(target_actions)
        dataset = TensorDataset(traces_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
        n_batches = 0

        for batch_traces, batch_actions in dataloader:
            batch_traces = batch_traces.to(device)
            batch_actions = batch_actions.to(device)

            self.optimizer.zero_grad()
            loss, loss_dict = self.compute_loss(batch_traces, batch_actions)
            loss.backward()
            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            n_batches += 1

        # Average over batches
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        return epoch_losses

    def encode(self, behavior_trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a behavior trace to latent Gaussian parameters.

        Args:
            behavior_trace: Behavior trace of shape (nS, 2) or (batch, nS, 2)

        Returns:
            Tuple of (mu, log_var), each of shape (latent_size,) or (batch, latent_size)
        """
        self.encoder.eval()

        with torch.no_grad():
            # Add batch dim if needed
            trace = np.array(behavior_trace)
            if trace.ndim == 2:
                trace = trace[np.newaxis, ...]
                squeeze = True
            else:
                squeeze = False

            trace_tensor = torch.FloatTensor(trace).to(device)
            mu, log_var = self.encoder(trace_tensor)

            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy()

            if squeeze:
                mu_np = mu_np.squeeze(0)
                log_var_np = log_var_np.squeeze(0)

            return mu_np, log_var_np

    def get_covariance(self, log_var: np.ndarray) -> np.ndarray:
        """Convert log_var to diagonal covariance matrix.

        Args:
            log_var: Log variance of shape (latent_size,)

        Returns:
            Diagonal covariance matrix of shape (latent_size, latent_size)
        """
        var = np.exp(log_var)
        return np.diag(var)

    def sample(self, mu: np.ndarray, log_var: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from the latent distribution.

        Args:
            mu: Mean of shape (latent_size,)
            log_var: Log variance of shape (latent_size,)
            n_samples: Number of samples to generate

        Returns:
            Samples of shape (n_samples, latent_size)
        """
        std = np.exp(0.5 * log_var)
        samples = np.random.randn(n_samples, len(mu)) * std + mu
        return samples

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent samples to action probabilities.

        Args:
            z: Latent samples of shape (latent_size,) or (batch, latent_size)

        Returns:
            Action probabilities of shape (n_states, n_actions) or (batch, n_states, n_actions)
        """
        self.decoder.eval()

        with torch.no_grad():
            z_arr = np.array(z)
            if z_arr.ndim == 1:
                z_arr = z_arr[np.newaxis, ...]
                squeeze = True
            else:
                squeeze = False

            z_tensor = torch.FloatTensor(z_arr).to(device)
            probs = self.decoder(z_tensor)
            probs_np = probs.cpu().numpy()

            if squeeze:
                probs_np = probs_np.squeeze(0)

            return probs_np

    def save(self, path: str) -> None:
        """Save VAE model to disk.

        Args:
            path: Base path for saving (without extension)
        """
        state_dict = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'config': {
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'hidden_size': self.hidden_size,
                'latent_size': self.latent_size,
                'learning_rate': self.learning_rate,
                'beta': self.beta
            }
        }
        torch.save(state_dict, f"{path}_vae.pth")

    def load(self, path: str) -> None:
        """Load VAE model from disk.

        Args:
            path: Base path for loading (without extension)
        """
        state_dict = torch.load(f"{path}_vae.pth", map_location=device, weights_only=True)
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])


def create_behavior_trace(states: np.ndarray, actions: np.ndarray, n_states: int) -> np.ndarray:
    """Create a behavior trace matrix from state-action pairs.

    A behavior trace is an (nS, 2) matrix where:
    - Column 0: state index
    - Column 1: last action taken in that state (4 if not visited)

    Args:
        states: Array of visited states
        actions: Array of actions taken
        n_states: Total number of states

    Returns:
        Behavior trace of shape (n_states, 2)
    """
    trace = np.ones((n_states, 2)) * 4  # 4 = unvisited action placeholder
    trace[:, 0] = np.arange(n_states)  # State indices

    for state, action in zip(states, actions):
        trace[int(state), 1] = action

    return trace


def generate_synthetic_behavior_traces(
    n_samples: int,
    n_states: int,
    n_actions: int,
    coverage_prob: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic behavior traces for VAE pre-training.

    Args:
        n_samples: Number of samples to generate
        n_states: Number of states
        n_actions: Number of actions
        coverage_prob: Probability of visiting each state

    Returns:
        Tuple of (behavior_traces, target_actions) for training
    """
    traces = np.ones((n_samples, n_states, 2)) * 4
    traces[:, :, 0] = np.arange(n_states)

    # Generate random actions for each state
    actions = np.random.randint(0, n_actions, size=(n_samples, n_states))

    # Apply random coverage mask
    coverage_mask = np.random.random((n_samples, n_states)) < coverage_prob

    for i in range(n_samples):
        for s in range(n_states):
            if coverage_mask[i, s]:
                traces[i, s, 1] = actions[i, s]

    return traces, actions


if __name__ == "__main__":
    # Test VAE
    n_states = 16
    n_actions = 4
    latent_size = 8

    # Create VAE
    vae = BehaviorVAE(
        n_states=n_states,
        n_actions=n_actions,
        hidden_size=128,
        latent_size=latent_size
    )

    # Generate synthetic data
    traces, actions = generate_synthetic_behavior_traces(
        n_samples=1000,
        n_states=n_states,
        n_actions=n_actions
    )

    print(f"Training data shape: traces={traces.shape}, actions={actions.shape}")

    # Train for a few epochs
    for epoch in range(5):
        losses = vae.train_epoch(traces, actions, batch_size=64)
        print(f"Epoch {epoch+1}: total={losses['total']:.4f}, recon={losses['recon']:.4f}, kl={losses['kl']:.4f}")

    # Test encoding
    test_trace = traces[0]
    mu, log_var = vae.encode(test_trace)
    print(f"\nEncoded trace: mu={mu.shape}, log_var={log_var.shape}")
    print(f"mu: {mu}")
    print(f"log_var: {log_var}")

    # Test covariance
    cov = vae.get_covariance(log_var)
    print(f"\nCovariance matrix shape: {cov.shape}")

    # Test sampling
    samples = vae.sample(mu, log_var, n_samples=5)
    print(f"\nSamples shape: {samples.shape}")

    # Test decoding
    decoded = vae.decode(samples[0])
    print(f"\nDecoded probabilities shape: {decoded.shape}")
    print(f"Sum of probs per state: {decoded.sum(axis=1)}")
