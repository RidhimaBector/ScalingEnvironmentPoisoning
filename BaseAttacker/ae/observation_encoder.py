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

import numpy as np

from core.encoder import Encoder


class ObservationEncoder:
    """Encodes victim results into observations for the attack algorithm.

    Wraps a VictimEncoder and exposes a clean encode(victim_data) interface.
    The caller (training loop) is responsible for obtaining victim_data from
    the environment step info and passing it here.

    Args:
        inner_encoder: VictimEncoder (or any Encoder ABC implementation).
    """

    def __init__(self, inner_encoder: Encoder, append_dynamics_dim: int = 0,
                 normalize_dynamics: bool = False, initial_dynamics=None):
        # append_dynamics_dim > 0: concatenate the env dynamics the attacker itself
        # controls (altitude = base + its own accumulated perturbations) to the
        # embedding, so a trajectory-only encoder still knows what it has done so far.
        self._inner = inner_encoder
        self._dyn_dim = append_dynamics_dim
        self._normalize = normalize_dynamics
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
            if self._normalize:
                dyn = (dyn - 5.0) / 5.0
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
                if self._normalize:
                    dyn = (dyn - 5.0) / 5.0
            emb = np.concatenate([np.asarray(emb, dtype=np.float64), dyn])
        return emb

    def train_encoder(self) -> None:
        self._inner.train_encoder()

    def compute_reward(self, victim_data: List[Dict], target) -> float:
        return self._inner.compute_reward(victim_data, target)
