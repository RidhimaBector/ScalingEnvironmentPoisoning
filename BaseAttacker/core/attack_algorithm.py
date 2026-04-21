"""Abstract base class for attack RL algorithms (DDPG, TD3, PPO, etc.).

The attack algorithm learns a policy that maps attack states (encoded victim
information + environment dynamics) to perturbation actions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class AttackAlgorithm(ABC):
    """ABC for attack RL algorithms in the environment poisoning framework.

    Implementations learn a continuous-action policy (e.g., DDPG, PPO)
    that outputs environment perturbations.

    Each algorithm owns its own replay/rollout buffer internally.
    The train loop calls: store_transition() -> ready_to_train() -> update().
    """

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """Select a perturbation action given the current attack state.

        Args:
            state: 1D numpy array of the current attack state.

        Returns:
            1D numpy array of the perturbation action.
        """

    @abstractmethod
    def store_transition(self, obs: np.ndarray, action: np.ndarray,
                         next_obs: np.ndarray, reward: float,
                         done: bool) -> None:
        """Store a single transition in the algorithm's internal buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            next_obs: Next observation.
            reward: Reward received.
            done: Whether the episode terminated.
        """

    @abstractmethod
    def ready_to_train(self) -> bool:
        """Return True when enough data has been collected to run update()."""

    @abstractmethod
    def update(self, episode: int = 0) -> Dict[str, float]:
        """Perform one or more training updates using internally stored data.

        Args:
            episode: Current episode number (for logging / scheduling).

        Returns:
            Dictionary of training metrics (e.g., critic_loss, actor_loss).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: Base path for saving (implementation adds suffixes).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: Base path to load from.
        """

    # ------------------------------------------------------------------
    # Concrete methods (overridable)
    # ------------------------------------------------------------------

    @property
    def is_on_policy(self) -> bool:
        """Whether the algorithm is on-policy (e.g., PPO). Default False."""
        return False

    @property
    def warmup_episodes(self) -> int:
        """Number of episodes to collect with random actions before training."""
        return 0

    def get_loss_log(self) -> list:
        """Return the full loss history. Default empty list."""
        return []

    def save_buffer(self, path: str) -> None:
        """Save the internal buffer/rollout to disk. Default no-op."""
        pass

    # ------------------------------------------------------------------
    # Deprecated — kept so old code doesn't crash on import
    # ------------------------------------------------------------------

    def train_step(self, replay_buffer: Any, batch_size: int,
                   **kwargs) -> Dict[str, float]:
        """Deprecated. Use store_transition/ready_to_train/update instead."""
        raise NotImplementedError(
            "train_step is deprecated. Use store_transition -> ready_to_train -> update."
        )
