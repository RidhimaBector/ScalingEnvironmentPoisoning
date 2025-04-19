from abc import ABC, abstractmethod
import numpy as np

class EnvironmentEncoderInterface(ABC):
    @abstractmethod
    def encode_state(self, environment_state) -> np.ndarray:
        """Encode a given environment state into embedding."""
        pass

    @abstractmethod
    def get_initial_state(self, environment_state) -> np.ndarray:
        """Get an initial embedding (e.g., all-zero or default embedding)"""
        pass
