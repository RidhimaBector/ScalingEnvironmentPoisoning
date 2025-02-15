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

        # self.altitude = self._generate_altitude(self.shape)

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        self.state = self.initial_state
        return self.state
    

    # def step(self, action):
    #     assert self.action_space.contains(action)
    #     next_state = self._calculate_next_state(self.state, action)
    #     reward = self._calculate_reward(next_state)
    #     done = self._is_terminal(next_state)
    #     self.state = next_state
    #     return next_state, reward, done, {}