from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from gym import Env, spaces


class Environment(Env):
    """Abstract base class for environments."""


    def __init__(self):
        self.seed()

        self.initial_state = np.zeros(2)
        self._state = self.initial_state
        self.metadata = {'render.modes': ['human', 'ansi']}


    @abstractmethod
    def seed(self, seed=None):
        """Seed the environment."""
        pass


    @abstractmethod
    def reset(self) -> Any:
        """Reset environment to initial state."""
        pass


    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """Take action in environment."""
        pass


    @abstractmethod
    def render(self) -> None:
        """Render environment state."""
        pass


    @property
    def state(self):
        return self._state


    @state.setter
    def state(self, value):
        self._state = value
