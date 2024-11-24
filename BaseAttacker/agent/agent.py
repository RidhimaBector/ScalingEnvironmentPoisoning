from typing import Tuple
from envs.environment import VictimEnvironment, AttackEnvironment, Environment
from algorithms.algorithm import Algorithm

from abc import ABC, abstractmethod

from utils.utils_attack import *

"""
Agent class
"""

class Agent(ABC):
    """
    Agent class
    """
    def __init__(self, env: Environment, algorithm: Algorithm):
        self.env = env
        self.algorithm = algorithm

    @abstractmethod
    def reset(self) -> Tuple[Environment, Algorithm]:
        """Reset the environment and algorithm"""
        pass


class AttackAgent(Agent):
    """
    AttackAgent class
    """
    def __init__(self, env: AttackEnvironment, algorithm: Algorithm):
        self.env = env
        self.algorithm = algorithm

    def reset(self) -> Tuple[AttackEnvironment, Algorithm]:
        """Reset the attack environment and algorithm"""
        self.env.reset()
        self.algorithm.reset()

        return self.env , self.algorithm


    def attack_effort(self, env_dynamics):
        return self.env.Attack_Effort(env_dynamics)


    def attack_cost_compute_K(self, target, cost_matrix):
        distance_k = Attack_Cost_Compute_K(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=0)
        distance_grid_k = Attack_Cost_Compute_K(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=1)
        distance_behavior_k = Attack_Cost_Compute_K(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=2)
        return -distance_k, -distance_grid_k, -distance_behavior_k


    def attack_cost_compute_W(self, target, cost_matrix):
        distance_w = Attack_Cost_Compute_W(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=0)
        distance_grid_w = Attack_Cost_Compute_W(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=1)
        distance_behavior_w = Attack_Cost_Compute_W(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=2)
        return -distance_w, -distance_grid_w, -distance_behavior_w


    def attack_done_identify(self, target):
        return Attack_Done_Identify(target, self.env.victim_algo.Q)


class VictimAgent(Agent):
    """
    VictimAgent class
    """

    """Will be used for the victim policy and environment"""

    def __init__(self, env: VictimEnvironment, algorithm: Algorithm):
        super().__init__(env, algorithm)


    def reset(self) -> Tuple[VictimEnvironment, Algorithm]:
        """Reset the victim environment and algorithm"""
        self.env.reset_altitude()
        self.algorithm.reset()

        return self.env , self.algorithm
