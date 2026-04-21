"""Abstract base class for encoders that produce attack state embeddings.

Encoders transform victim data (Q-tables, behavior traces, trajectories)
into fixed-size embeddings used as the attack agent's observation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class Encoder(ABC):
    """ABC for encoders in the attack framework.

    Encoders consume a list of per-victim result dictionaries and produce
    a single fixed-size embedding. Both single-victim and collective
    (multi-victim) encoders implement this interface.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the embedding produced by encode().

        Returns:
            Integer embedding dimension.
        """

    @abstractmethod
    def encode(self, victim_data: List[Dict], env_dynamics: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode victim data into a fixed-size embedding.

        Args:
            victim_data: List of dicts, one per victim. Each dict may contain:
                - 'policy_matrix': (nS, nA) Q-table or policy (whitebox)
                - 'behavior_trace': (nS, 2) last-action-per-state (blackbox)
                - 'trajectories': list of (s, a, r, s') tuples (partial blackbox)
                Other keys depend on the specific encoder.
            env_dynamics: Optional environment dynamics array. Whitebox
                encoders may include this in the embedding; blackbox
                encoders ignore it.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """

    @abstractmethod
    def get_initial_embedding(self) -> np.ndarray:
        """Return a zero or default embedding for the initial state.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
