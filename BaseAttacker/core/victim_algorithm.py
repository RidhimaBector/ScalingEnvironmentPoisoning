"""Abstract base class for victim RL algorithms.

Any RL algorithm that can serve as a victim (Q-learning, DQN, PPO, A2C, SARSA)
should implement this interface to be usable in the attack framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class VictimAlgorithm(ABC):
    """ABC for victim RL algorithms used in environment poisoning attacks.

    Implementations must provide methods for training, acting, and exposing
    policy information at varying levels of detail (for different privacy modes).
    """

    @abstractmethod
    def train(self, env: Any, num_episodes: int) -> Dict[str, Any]:
        """Train the algorithm on the given environment.

        Args:
            env: Environment to train on (VictimEnvironment instance).
            num_episodes: Number of training episodes.

        Returns:
            Dictionary with training statistics (rewards, lengths, etc.).
        """

    @abstractmethod
    def act(self, state: Any) -> int:
        """Select an action given the current state.

        Args:
            state: Current environment state.

        Returns:
            Selected action index.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the algorithm to its initial untrained state."""

    @abstractmethod
    def get_policy_matrix(self) -> np.ndarray:
        """Return the policy as a (nS, nA) matrix.

        For Q-learning: the Q-table.
        For policy gradient: action probabilities per state.

        Returns:
            Policy matrix of shape (nS, nA).
        """

    @abstractmethod
    def get_behavior_trace(self) -> np.ndarray:
        """Return the behavior trace: last observed action per state.

        The trace is an (nS, 2) matrix where column 0 is the state index
        and column 1 is the last greedy action taken in that state
        (-1 = unvisited).

        Returns:
            Behavior trace of shape (nS, 2).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save algorithm state to disk.

        Args:
            path: Base path for saving (implementation adds extensions).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load algorithm state from disk.

        Args:
            path: Base path to load from.
        """

    def get_whitebox_data(self) -> Dict[str, Any]:
        """Return internal algorithm state exposed in whitebox privacy mode.

        Override to expose algorithm-specific internals (e.g. value functions,
        visit counts, model parameters).  The default returns the policy matrix.

        Returns:
            Dict whose values are numpy arrays.  Standard key:
                'policy_matrix': the (nS, nA) policy or Q-table.
        """
        return {'policy_matrix': self.get_policy_matrix().copy()}

    def get_greedy_actions(self) -> np.ndarray:
        """Return the greedy action for each state.

        Returns:
            Array of shape (nS,) with greedy action indices.
        """
        return np.argmax(self.get_policy_matrix(), axis=1)

    def get_trajectories(self) -> list:
        """Return recent (pre_state, pre_action, state, action) tuples and clear the buffer.

        Override in subclasses to collect per-step transitions during training.
        The LSTM trajectory encoder reads this key from victim_data.

        Returns:
            List of (pre_state, pre_action, state, action) 4-tuples.
            pre_state / state: raw env observation (int for Discrete, array for Box).
            pre_action / action: int action index.
        """
        return []
