"""Abstract base class for victim environments that support poisoning.

Any environment that can be poisoned (Grid3D, LunarLander, etc.) should
implement this interface to define how perturbations are applied and
how dynamics are exposed.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
from gymnasium import Env, spaces


class TabularModel(NamedTuple):
    """An environment with discrete states, written out as a table. Environments that
    can return one from tabular_model() let tabular victims train with compiled loops
    (victim/fast_tabular.py), which is many times faster; it changes nothing else."""

    cdf: np.ndarray  # (nA, nS, nS) cumulative transition probabilities over next states
    start: int  # state every episode starts in
    terminal: np.ndarray  # (nS,) bool: arriving here ends the episode
    reward: np.ndarray  # (nS,) reward for arriving in each state
    max_steps: int  # episode step limit


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

    def dynamics_bounds(self):
        """(low, high) of the values in get_dynamics(), or None if unbounded/unknown.

        Used only to scale the dynamics vector when it is part of the attacker's
        observation. Environments without natural bounds can leave the default.
        """
        return None

    def make_target_policy(self, path_type: str = "Mp"):
        """Target policy matrix (nS, nA) the attacker should induce, or None.

        Rows are states; the target action has a 1, every other entry 0; rows of
        states the attacker does not care about are all zero. Return None if the
        target is given by target_trajectories() instead.
        """
        return None

    def target_trajectories(self):
        """The target as recorded behaviour: a list of (prev_state, prev_action, state,
        action) tuples, or None. Use this instead of make_target_policy() when states
        are continuous and no per-state table exists."""
        return None

    def tabular_model(self) -> Optional[TabularModel]:
        """The current dynamics as a TabularModel, if this environment is a finite MDP
        with one start state. Optional: None (the default) means victims step through
        reset()/step() as usual."""
        return None

    def oracle_route_match(self, target_actions: np.ndarray, greedy_actions: np.ndarray):
        """Evaluation only: 1.0 if following greedy_actions from the start reproduces the
        target route under the current dynamics, else 0.0. None if the environment has
        no notion of a route. Never shown to the attacker."""
        return None

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
