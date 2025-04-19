import numpy as np
import torch
from gym import spaces
from typing import TYPE_CHECKING, Tuple

from algorithms.algorithm import Algorithm
from constants import *
from envs.environment import Environment
from ae.encoder_service import EncoderService

if TYPE_CHECKING:
    from agent.victim_system import VictimSystem

class AttackEnvironment(Environment):
    def __init__(self, victim_system: 'VictimSystem', config):
        super(AttackEnvironment, self).__init__()
        self._victim_system = victim_system
        self.nS = EMBEDDING_SIZE + self.victim_system.env.nS
        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.victim_system.env.nS,), dtype=np.float64)
        self.observation_space = spaces.Discrete(self.nS + 1)
        self.is_initial_state = True

    def train_victim(self, num_episodes: int):
        self.victim_system.algorithm.train(
            env=self.victim_system.env,
            num_episodes=num_episodes
        )

    def reset(self):
        self.is_initial_state = True
        self.victim_system.reset()
        return True

    def step(self, U: torch.Tensor) -> Tuple[torch.Tensor, bool, dict, dict, list, list]:
        def Attack_Env(self, U: torch.Tensor) -> Tuple[torch.Tensor, bool, dict, dict, list, list]:
            A = np.clip(self.victim_system.env.altitude + U.reshape(self.victim_system.env.shape), 0.0, 10.0)
            self.victim_system.env.altitude = A
            self.victim_system.env.T = self.victim_system.env._calculate_dynamics(self.victim_system.env.shape, self.victim_system.env.nA, self.victim_system.env.nS, A)
            return 0

        return Attack_Env(self, U)

    def Attack_Effort(self, env_dynamics):
        prev_env_dynamics = env_dynamics.copy()
        current_env_dynamics_updated = self.victim_system.env.env_dynamics.copy()
        action_clipped = current_env_dynamics_updated - prev_env_dynamics
        effort = np.mean(np.abs(action_clipped))
        return effort, current_env_dynamics_updated

    @property
    def victim_system(self) -> 'VictimSystem':
        return self._victim_system

    @victim_system.setter
    def victim_system(self, value: 'VictimSystem'):
        self._victim_system = value


