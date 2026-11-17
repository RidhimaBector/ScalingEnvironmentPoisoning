"""ObservationEncoder: transforms victim results into attack observations.

A pure transformation service.  The training loop calls encode(victim_data)
with the results returned by AttackEnvironment.step(), and uses the output
as the observation passed to the attack algorithm.

Usage:
    obs_encoder = ObservationEncoder(VictimEncoder.from_population(population))

    obs = obs_encoder.get_initial_embedding()
    for t in range(max_timesteps):
        _, reward, done, _, info = attack_env.step(dispatch_action)
        next_obs = obs_encoder.encode(info['victim_results'])
        algo.store_transition(obs, action, next_obs, reward, done)
        obs = next_obs
"""

from typing import Dict, List

import os

import numpy as np

from core.encoder import Encoder


def _scale(x, bounds):
    """Map x from [low, high] to [-1, 1]; unchanged if bounds is None."""
    if bounds is None:
        return x
    low, high = bounds
    return (np.asarray(x, dtype=np.float64) - (low + high) / 2.0) / ((high - low) / 2.0)


class ObservationEncoder:
    """Encodes victim results into observations for the attack algorithm.

    Wraps a VictimEncoder and exposes a clean encode(victim_data) interface.
    The caller (training loop) is responsible for obtaining victim_data from
    the environment step info and passing it here.

    Args:
        inner_encoder: VictimEncoder (or any Encoder ABC implementation).
    """

    def __init__(self, inner_encoder: Encoder, append_dynamics_dim: int = 0,
                 normalize_dynamics: bool = False, initial_dynamics=None, dynamics_bounds=None):
        # append_dynamics_dim > 0: concatenate the env dynamics the attacker itself
        # controls (altitude = base + its own accumulated perturbations) to the
        # embedding, so a trajectory-only encoder still knows what it has done so far.
        self._inner = inner_encoder
        self._dyn_dim = append_dynamics_dim
        # Scale the appended vector to [-1, 1] using the environment's own bounds.
        self._bounds = dynamics_bounds if normalize_dynamics else None
        self._initial_dynamics = None if initial_dynamics is None else np.asarray(initial_dynamics).flatten()

    @property
    def embedding_dim(self) -> int:
        return self._inner.embedding_dim + self._dyn_dim

    def encode(self, victim_data: List[Dict], env_dynamics=None) -> np.ndarray:
        """Encode victim result dicts into a 1D observation array.

        Args:
            victim_data: List of per-victim result dicts from
                AttackEnvironment.step() info['victim_results'].
            env_dynamics: Optional env dynamics array (e.g. altitude).
                Passed through to the inner encoder; ignored by VictimEncoder.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        emb = self._inner.encode(victim_data, env_dynamics=env_dynamics)
        if self._dyn_dim:
            dyn = np.zeros(self._dyn_dim) if env_dynamics is None else np.asarray(env_dynamics).flatten()[:self._dyn_dim]
            dyn = _scale(dyn, self._bounds)
            emb = np.concatenate([np.asarray(emb, dtype=np.float64), dyn])
        return emb

    def get_initial_embedding(self) -> np.ndarray:
        """Return a zero embedding for the start of an episode.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        emb = self._inner.get_initial_embedding()
        if self._dyn_dim:
            dyn = np.zeros(self._dyn_dim)
            if self._initial_dynamics is not None:
                dyn = self._initial_dynamics[:self._dyn_dim]
                dyn = _scale(dyn, self._bounds)
            emb = np.concatenate([np.asarray(emb, dtype=np.float64), dyn])
        return emb

    def target_similarity(self) -> float:
        """Similarity of victim behaviour to the target, if the inner encoder provides it."""
        fn = getattr(self._inner, 'target_similarity', None)
        return float(fn()) if fn is not None else 0.0

    def reset_episode(self) -> None:
        fn = getattr(self._inner, 'reset_episode', None)
        if fn is not None:
            fn()

    def save(self, path: str) -> None:
        """Save the inner encoder if it is trained online (no-op for fixed encoders)."""
        fn = getattr(self._inner, 'save', None)
        if fn is not None:
            fn(path)

    def load(self, path: str) -> None:
        fn = getattr(self._inner, 'load', None)
        if fn is not None and os.path.exists(path):
            fn(path)

    def train_encoder(self) -> None:
        self._inner.train_encoder()

    @property
    def warmup_episodes(self) -> int:
        """Exploration-only attack episodes the inner encoder needs before fit()."""
        return int(getattr(self._inner, 'warmup_episodes', 0))

    def fit(self) -> float:
        """Fit and freeze an encoder that learns from the run's first episodes."""
        return float(self._inner.fit())

    def compute_reward(self, victim_data: List[Dict], target) -> float:
        return self._inner.compute_reward(victim_data, target)
