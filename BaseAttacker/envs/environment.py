import numpy as np
import math
from gym import spaces, Env
from gym.utils import seeding
from constants import *
import torch


class Environment(Env):
    metadata = {'render.modes': ['human', 'ansi']}


    def __init__(self):
        # self.shape = (4, 4)
        # self.nS = np.prod(self.shape)
        
        # self.observation_space = spaces.Discrete(self.nS)
        self._seed()

        self.initial_state = np.zeros(2)
        self.state = self.initial_state

        self.altitude = self._generate_altitude(self.shape)

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        self.state = self.initial_state
        return self.state
    

    def step(self, action):
        assert self.action_space.contains(action)
        next_state = self._calculate_next_state(self.state, action)
        reward = self._calculate_reward(next_state)
        done = self._is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done, {}
    

class AttackerEnv(Environment):

    def __init__(self, victim, VictimEnv):
        super(AttackerEnv, self).__init__()
        self.victim = victim
        self.victim_env = VictimEnv

        # self.nA = spaces.Box(low=-1.0, high=+1.0, shape=(self.nS,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.nS,), dtype=np.float64)
        self.observation_space = spaces.Discrete(self.nS + 1)
        # self._seed()
        self.nS = EMBEDDING_SIZE + self.victim_env.nS

        self.initial_state = self.get_initial_state()
        self.state = self.initial_state

        # self.altitude = self._generate_altitude(self.shape)


    def get_initial_state(self):
        # Initialize the attacker's state
        victim_info = np.zeros((1,EMBEDDING_SIZE))
        victim_tensor = torch.from_numpy(victim_info)
        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

        env_info = self.victim_env.altitude.copy()
        env_tensor = torch.from_numpy(env_info)
        env_tensor = env_tensor.view(1, self.victim_env.nS)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        return torch.cat((victim_tensor_4d, env_tensor_4d), 3)


    def get_next_state(self, auto_encoder, victim_transitions):

        victim_info = auto_encoder.Policy_Embedding(victim_transitions)
        victim_tensor = torch.from_numpy(victim_info[-1]).unsqueeze(0)
        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

        env_info = self.victim_env.altitude.copy() #system.victim.env.altitude.copy()
        env_tensor = torch.from_numpy(env_info)
        env_tensor = env_tensor.view(1, self.victim_env.nS)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        return torch.cat((victim_tensor_4d, env_tensor_4d), 3)


    def step(self, U):
        def Attack_Env(self, U):
            A = np.clip(self.altitude + U.reshape(self.shape), 0.0, 10.0)

            self.altitude = A
            self.T = self._calculate_dynamics(self.shape, self.nA, self.nS, A)
            return 0
        
        return Attack_Env(self, U)
        

    def update_victim(self, victim):
        self.victim = victim
        
    
    def update_victim_env(self, victim_env):
        self.victim_env = victim_env
