from typing import Tuple

from agent.system import System
from algorithms.algorithm import Algorithm
from envs.victim_environment import VictimEnvironment

class VictimSystem(System):
    """
    VictimSystem class that manages victim agent training and evaluation
    """
    def __init__(self, env: VictimEnvironment, algorithm: Algorithm):
        super().__init__(env, algorithm)

    def reset(self) -> Tuple[VictimEnvironment, Algorithm]:
        """Reset the victim environment and algorithm"""
        self.env.reset_altitude()
        self.algorithm.reset()

        return self.env, self.algorithm
