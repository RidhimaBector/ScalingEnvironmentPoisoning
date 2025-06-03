import numpy as np
import torch
from gym import spaces
from typing import TYPE_CHECKING

from algorithms.algorithm import Algorithm
from constants import *
from envs.environment import Environment
from envs.victim_environment import VictimEnvironment
from agent.victim_system import VictimSystem
from ae.encoder_service import EncoderService

class AttackEnvironment(Environment):
    def __init__(self, victim_system: VictimSystem):
        super(AttackEnvironment, self).__init__()
        self._victim_env: VictimEnvironment = victim_system.env
        self._victim_algorithm: Algorithm = victim_system.algorithm
        self.victim_system = victim_system
        self.nS = EMBEDDING_SIZE + self.victim_env.nS
        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.victim_env.nS,), dtype=np.float64)
        self.observation_space = spaces.Discrete(self.nS + 1)
        self.is_initial_state = True

    def get_state(self, encoder_service: EncoderService):
        """Get combined state representation from policy and environment."""
        if self.is_initial_state:
            # Initialize the attacker's state
            # victim_info = np.zeros((1,EMBEDDING_SIZE))
            # victim_tensor = torch.from_numpy(victim_info)
            # victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

            # env_info = env.altitude.copy()
            # env_tensor = torch.from_numpy(env_info)
            # env_tensor = env_tensor.view(1, env.nS)
            # env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

            # x = torch.cat((victim_tensor_4d, env_tensor_4d), 3)
            # curA = env.altitude.copy().reshape((16, 1))

            return encoder_service.get_initial_state()

        transitions = self.victim_algo.transitions
        return encoder_service.encode_state(
            self.victim_env.altitude,
            transitions
        )

    def reset(self):
        self.is_initial_state = True
        self.victim_system.reset()
        return True

    def step(self, U):
        def Attack_Env(self, U):
            A = np.clip(self._victim_env.altitude + U.reshape(self._victim_env.shape), 0.0, 10.0)
            self._victim_env.altitude = A
            self._victim_env.T = self._victim_env._calculate_dynamics(self._victim_env.shape, self._victim_env.nA, self._victim_env.nS, A)
            return 0

        return Attack_Env(self, U)

    def Attack_Effort(self, env_dynamics):
        prev_env_dynamics = env_dynamics.copy()
        current_env_dynamics_updated = self.victim_env.env_dynamics.copy()
        action_clipped = current_env_dynamics_updated - prev_env_dynamics
        effort = np.mean(np.abs(action_clipped))
        return effort, current_env_dynamics_updated

    @property
    def victim_algorithm(self) -> Algorithm:
        return self._victim_algorithm

    @victim_algorithm.setter
    def victim_algorithm(self, value: Algorithm):
        self._victim_algorithm = value

    @property
    def victim_env(self) -> Environment:
        return self._victim_env

    @victim_env.setter
    def victim_env(self, value):
        self._victim_env = value
