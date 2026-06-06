"""Abstract base class for victim environments that support poisoning.

Any environment that can be poisoned (Grid3D, LunarLander, etc.) should
implement this interface to define how perturbations are applied and
how dynamics are exposed.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

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

    def compute_distance_metrics(self, algo: Any, target: np.ndarray) -> Dict[str, float]:
        """Distance from victim's current policy to the target policy.

        Override in concrete environments that support distance metrics
        (e.g. Wasserstein distance rate, KL-rate). Environments that don't
        support these metrics return an empty dict.

        Args:
            algo: Victim algorithm (provides get_policy_matrix()).
            target: Target policy matrix of shape (nS, nA).

        Returns:
            Dict of metric_name -> scalar float.
        """
        return {}

    def compute_effort(self) -> float:
        """Mean absolute change in dynamics since last apply_perturbation().

        Relies on subclasses setting self._prev_dynamics before updating
        dynamics in apply_perturbation(). Returns 0.0 if no perturbation
        has been applied yet.

        Returns:
            Scalar effort value.
        """
        if hasattr(self, '_prev_dynamics') and self._prev_dynamics is not None:
            return float(np.mean(np.abs(self.get_dynamics() - self._prev_dynamics)))
        return 0.0
