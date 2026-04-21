"""Abstract base class for victim environments that support poisoning.

Any environment that can be poisoned (Grid3D, LunarLander, etc.) should
implement this interface to define how perturbations are applied and
how dynamics are exposed.
"""

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import Env, spaces


class VictimEnvironment(Env, ABC):
    """ABC for environments that support environment poisoning attacks.

    Subclasses must define how perturbations modify environment dynamics
    and expose the current dynamics for use in the attack state.
    """

    @property
    @abstractmethod
    def perturbation_space(self) -> spaces.Space:
        """The space of valid perturbation vectors.

        Returns:
            gym.spaces.Space describing valid perturbations.
        """

    @abstractmethod
    def apply_perturbation(self, perturbation: np.ndarray) -> None:
        """Apply an attack perturbation to modify environment dynamics.

        Args:
            perturbation: Perturbation vector matching perturbation_space.
        """

    @abstractmethod
    def get_dynamics(self) -> np.ndarray:
        """Return the current environment dynamics as a flat array.

        For Grid3D: altitude flattened to (nS,).
        For LunarLander: gravity + wind parameters.

        Returns:
            1D numpy array of current dynamics.
        """

    @abstractmethod
    def reset_dynamics(self) -> None:
        """Reset environment dynamics to their original (unpoisoned) values."""

    @property
    @abstractmethod
    def dynamics_dim(self) -> int:
        """Dimensionality of the dynamics vector returned by get_dynamics().

        Returns:
            Integer dimension.
        """

    def get_whitebox_data(self) -> dict:
        """Return internal environment state exposed in whitebox privacy mode.

        Override to expose environment-specific internals (e.g. full transition
        matrices, reward functions, model parameters).  The default returns the
        dynamics vector.

        Returns:
            Dict whose values are numpy arrays.  Standard key:
                'env_dynamics': the current dynamics as a flat array.
        """
        return {'env_dynamics': self.get_dynamics().flatten()}
