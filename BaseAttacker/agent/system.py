from abc import ABC, abstractmethod
from typing import Tuple

from algorithms.algorithm import Algorithm
from envs.environment import Environment

class System(ABC):
    """
    Abstract base System class
    """
    def __init__(self, env: Environment, algorithm: Algorithm):
        self._env = env
        self._algorithm = algorithm

    @property
    def env(self) -> Environment:
        return self._env

    @env.setter
    def env(self, value: Environment):
        self._env = value

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: Algorithm):
        self._algorithm = value

    @abstractmethod
    def reset(self) -> Tuple[Environment, Algorithm]:
        """Reset the environment and algorithm"""
        pass
