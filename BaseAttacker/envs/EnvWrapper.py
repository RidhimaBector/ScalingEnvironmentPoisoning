import gym
from gym import spaces
import numpy as np

class EnvWrapper(gym.Env): # INterface
    def __init__(self, env, observation_filter=None, action_filter=None):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # self.reward_range = env.reward_range
        self.oberservation_filter = observation_filter
        self.action_filter = action_filter
        self.Attack_ActionSpace = env.Attack_ActionSpace
        self.altidude = env.altidude
        self.altidude_default = env.altidude_defaultz   
    
    def reset(self):
        state = self.env.reset()
        ### Encoder eventually
        limited_state = self._limit_observation(state)
        return limited_state
    
    def step(self, action): #victim learning against victim env, collect info and report
        filtered_action = self._filter_action(action)
        next_state, reward, done, info = self.env.step(filtered_action)
        limited_next_state = self._limit_observation(next_state)
        return limited_next_state, reward, done, info
    
    # def _limit_observation(self, state):
    #     if self.oberservation_filter is not None:
    #         return self.oberservation_filter(state)
    #     return state
    
    # def _filter_action(self, action):
    #     if self.action_filter is not None:
    #         return self.action_filter(action)
    #     return action
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    # def render(self, mode='human'):
    #     return self.env.render(mode)
    
    # def close(self):
    #     return self.env.close()

    def Attack_Env(self, U):
        if hasattr(self.env, 'Attack_Env'):
            return self.env.Attack_Env(U)
        
    def reset_altitude(self):
        if hasattr(self.env, 'reset_altitude'):
            return self.env.reset_altitude()  

    def _calculate_dynamics(self, shape, nA, nS, altitude):
        if hasattr(self.env, '_calculate_dynamics'):
            return self.env._calculate_dynamics(shape, nA, nS, altitude)

    def _calculate_transition_status(self, s, action, T):
        if hasattr(self.env, '_calculate_transition_status'):
            return self.env._calculate_transition_status(s, action, T)

    def _defined_altitude(self, shape):
        if hasattr(self.env, '_defined_altitude'):
            return self.env._defined_altitude(shape)  
        
def observation_filter(state):
    # Encoder?
    # if isinstance(state, np.ndarray):
    #     limited_state = np.zeros_like(state)
    #     return limited_state

    # Return a representation of the state for the attack, attack, decode.
    return state

def action_filter(action):
    # enc?
    # if isinstance(action, np.ndarray):
    #     limited_action = np.zeros_like(action)
    #     limited_action = np.clip(limited_action, -1, 1)
    #     return limited_action

    # Return a representation of the action for the attack, attack, decode.
    return action