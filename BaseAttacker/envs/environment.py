import numpy as np
import math
from gym import spaces, Env
from gym.utils import seeding
from constants import *
import torch
from algorithms.algorithm import Algorithm
from ae.ae import AutoEncoder
from abc import abstractmethod
from typing import Tuple, Any, Dict

class Environment(Env):
    """Abstract base class for environments."""


    def __init__(self):
        # self.shape = (4, 4)
        # self.nS = np.prod(self.shape)

        # self.observation_space = spaces.Discrete(self.nS)
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


class AttackEnvironment(Environment):


    def __init__(self, VictimEnv: Environment, VictimAlgo: Algorithm):
        super(AttackEnvironment, self).__init__()
        self._victim_env = VictimEnv
        self._victim_algo = VictimAlgo
        self.nS = EMBEDDING_SIZE + self.victim_env.nS
        # self.nA = spaces.Box(low=-1.0, high=+1.0, shape=(self.nS,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.victim_env.nS,), dtype=np.float64)
        self.observation_space = spaces.Discrete(self.nS + 1)
        # self._seed()

        self.is_initial_state = True

        # self.altitude = self._generate_altitude(self.shape)


    def get_state(self, auto_encoder: AutoEncoder):
        if self.is_initial_state:
            victim_info = np.zeros((1, EMBEDDING_SIZE))
            victim_tensor = torch.from_numpy(victim_info)
            self.is_initial_state = False  # Reset flag after first use
        else:
            # Use the same logic as get_next_state for non-initial states
            victim_transitions = self.victim_algo.transitions
            victim_info = auto_encoder.Policy_Embedding(victim_transitions)
            victim_tensor = torch.from_numpy(victim_info[-1]).unsqueeze(0)

        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

        env_info = self.victim_env.altitude.copy()
        env_tensor = torch.from_numpy(env_info)
        env_tensor = env_tensor.view(1, self.victim_env.nS)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        return torch.cat((victim_tensor_4d, env_tensor_4d), 3)


    def reset(self):
        self.is_initial_state = True
        return True

    def step(self, U):
        def Attack_Env(self, U):
            A = np.clip(self.victim_env.altitude + U.reshape(self.victim_env.shape), 0.0, 10.0)

            self.victim_env.altitude = A
            self.victim_env.T = self.victim_env._calculate_dynamics(self.victim_env.shape, self.victim_env.nA, self.victim_env.nS, A)
            return 0

        return Attack_Env(self, U)

    def Attack_Effort(self, env_dynamics):
        prev_env_dynamics = env_dynamics.copy() #New
        current_env_dynamics_updated = self.victim_env.env_dynamics.copy()
        action_clipped = current_env_dynamics_updated - prev_env_dynamics #Size: 16x1
        effort = np.mean(np.abs(action_clipped))

        return effort, current_env_dynamics_updated

    @property
    def victim_algo(self) -> Algorithm:
        return self._victim_algo


    @victim_algo.setter
    def victim_algo(self, value):
        self._victim_algo = value


    @property
    def victim_env(self) -> Environment:
        return self._victim_env


    @victim_env.setter
    def victim_env(self, value):
        self._victim_env = value


class VictimEnvironment(Environment):
    pass
