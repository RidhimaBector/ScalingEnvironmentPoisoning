

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from core.encoder import Encoder

if TYPE_CHECKING:
    from agent.victim_system import VictimSystem


class VictimEncoder(Encoder):
    """Packs victim result dicts into a fixed-size embedding.

    Args:
        policy_dim: Flat size of the policy/Q-table data (from algo.get_whitebox_data).
        dynamics_dim: Flat size of the environment dynamics data (from env.get_whitebox_data).
        trace_dim: Flat size of the behavior trace (nS × 2).
        num_victims: Number of victim agents (K).
    """

    def __init__(
        self,
        policy_dim: int,
        dynamics_dim: int,
        trace_dim: int,
        num_victims: int = 1,
    ):
        self._policy_dim = policy_dim
        self._dynamics_dim = dynamics_dim
        self._trace_dim = trace_dim
        self._num_victims = num_victims
        self._per_victim_dim = policy_dim + dynamics_dim + trace_dim

    @classmethod
    def from_population(cls, population: 'VictimSystem') -> 'VictimEncoder':
        """Create a VictimEncoder with dimensions derived from the population.

        Queries the first victim's algo and env to determine the exact sizes
        of all whitebox fields, so nothing is hardcoded per algorithm or env.

        Args:
            population: VictimSystem with at least one victim initialised.

        Returns:
            VictimEncoder configured for this population.
        """
        ref_algo = population.algorithm
        ref_env = population.env

        # Policy dim: sum of all policy values in the algo's whitebox data
        algo_data = ref_algo.get_whitebox_data()
        policy_dim = sum(
            np.asarray(v).flatten().size
            for v in algo_data.values()
        )

        # Dynamics dim: from the environment ABC
        dynamics_dim = ref_env.dynamics_dim

        # Behavior trace is always (nS, 2)
        trace_dim = population.nS * 2

        return cls(
            policy_dim=policy_dim,
            dynamics_dim=dynamics_dim,
            trace_dim=trace_dim,
            num_victims=population.num_victims,
        )

    @property
    def embedding_dim(self) -> int:
        return self._per_victim_dim * self._num_victims

    def encode(self, victim_data: List[Dict], env_dynamics: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode victim result dicts into a flat embedding.

        Concatenates all values from 'whitebox_data' and 'env_data' keys when
        present, then appends the behavior trace.  Missing sections are zero-
        padded to maintain the fixed embedding size.

        Args:
            victim_data: List of per-victim result dicts.
            env_dynamics: Unused (kept for Encoder ABC compatibility).

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        parts = []
        for i in range(self._num_victims):
            data = victim_data[i] if i < len(victim_data) else {}

            # Collect all whitebox values except the reserved keys
            _reserved = {'behavior_trace', 'victim_idx', 'env_dynamics'}
            policy_values = [
                np.asarray(v).flatten()
                for k, v in data.items()
                if k not in _reserved
            ]
            policy_vec = (
                np.concatenate(policy_values)
                if policy_values
                else np.zeros(self._policy_dim)
            )
            policy_vec = _pad_or_clip(policy_vec, self._policy_dim)

            # Environment dynamics
            dyn = np.asarray(
                data.get('env_dynamics', np.zeros(self._dynamics_dim))
            ).flatten()
            dyn = _pad_or_clip(dyn, self._dynamics_dim)

            # Behavior trace — always present
            trace = np.asarray(
                data.get('behavior_trace', np.zeros((self._trace_dim,)))
            ).flatten()
            trace = _pad_or_clip(trace, self._trace_dim)

            parts.append(np.concatenate([policy_vec, dyn, trace]))

        return np.concatenate(parts)

    def get_initial_embedding(self) -> np.ndarray:
        return np.zeros(self.embedding_dim)

    def compute_reward(self, victim_data: List[Dict], target: np.ndarray) -> float:
        from utils.utils_attack import Attack_Done_Identify
        accuracies = []
        for data in victim_data:
            q = data.get('policy_matrix')
            if q is not None:
                _, acc, _, _ = Attack_Done_Identify(target.copy(), np.asarray(q))
                accuracies.append(acc)
        if not accuracies:
            return super().compute_reward(victim_data, target)
        return float(np.mean(accuracies))


def _pad_or_clip(arr: np.ndarray, target: int) -> np.ndarray:
    """Clip or zero-pad a 1-D array to exactly target elements."""
    if len(arr) >= target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)))
