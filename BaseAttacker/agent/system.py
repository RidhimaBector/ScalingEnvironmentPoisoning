from abc import ABC, abstractmethod
from typing import Tuple

from algorithms.algorithm import Algorithm
from envs.environment import Environment

class System(ABC):
    """
    Abstract base System class
    """
    def __init__(self, env: Environment, algorithm: Algorithm):
        self.env = env
        self.algorithm = algorithm

    @property
    def env(self) -> Environment:
        return self.env

    @property
    def algorithm(self) -> Algorithm:
        return self.algorithm

    @abstractmethod
    def reset(self) -> Tuple[Environment, Algorithm]:
        """Reset the environment and algorithm"""
        pass
