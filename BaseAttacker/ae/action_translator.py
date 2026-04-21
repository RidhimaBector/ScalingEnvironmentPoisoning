
import numpy as np
from gymnasium import spaces


class ActionTranslator:
    """Clips and scales a raw algorithm action to a dispatchable perturbation.

    Args:
        low: Lower bound array for the perturbation space.
        high: Upper bound array for the perturbation space.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self._low = np.asarray(low, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)

    @classmethod
    def from_space(cls, space: spaces.Box) -> 'ActionTranslator':
        """Construct from a gymnasium Box space.

        Args:
            space: Box action space describing the perturbation bounds.

        Returns:
            ActionTranslator configured to the space's bounds.
        """
        return cls(low=space.low.copy(), high=space.high.copy())

    def translate(self, action: np.ndarray) -> np.ndarray:
        """Clip action to the perturbation space bounds.

        Args:
            action: Raw action from the attack algorithm, shape (n,).

        Returns:
            1D numpy array clipped to [low, high].
        """
        return np.clip(action, self._low, self._high)
