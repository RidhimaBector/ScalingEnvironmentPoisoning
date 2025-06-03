"""Base interface for Reinforcement Learning Algorithms.

This interface defines the minimal set of methods that any RL algorithm
should implement to ensure consistency across the codebase.


"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Algorithm(ABC):
    """Abstract base class for all RL algorithms."""

    @abstractmethod
    def train(self, num_episodes: int) -> Dict[str, Any]:
        """Train the algorithm for a specified number of episodes.

        Args:
            num_episodes: Number of episodes to train

        Returns:
            Dict containing training statistics
        """
        pass

    @abstractmethod
    def act(self, state: Any) -> int:
        """Select an action given the current state.

        Args:
            state: Current environment state

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool) -> None:
        """Update the algorithm's parameters based on experience.

        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Resulting state
            done: Whether episode terminated
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the algorithm to initial state."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save algorithm state.

        Args:
            path: Path to save location
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load algorithm state.

        Args:
            path: Path to load from
        """
        pass
