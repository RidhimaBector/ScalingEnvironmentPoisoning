from typing import Tuple
import numpy as np
import torch
from agent.system import System
from algorithms.algorithm import Algorithm
from envs.victim_environment import VictimEnvironment
from enum import Enum
from ae.encoder_service import EncoderService, EncoderType


class PrivacyMode(Enum):
    FULL_WHITEBOX = 0
    PARTIAL_BLACKBOX = 1
    FULL_BLACKBOX = 2



class VictimSystem(System):
    """
    VictimSystem class that manages victim agent training and evaluation
    """
    def __init__(self, env: VictimEnvironment, algorithm: Algorithm, config, privacy_mode: PrivacyMode = PrivacyMode.FULL_WHITEBOX):
        super().__init__(env, algorithm)
        self.privacy_mode = privacy_mode

        # Initialize encoder based on privacy mode
        if privacy_mode == PrivacyMode.FULL_WHITEBOX:
            encoder_type = EncoderType.COMBINED
        elif privacy_mode == PrivacyMode.PARTIAL_BLACKBOX:
            encoder_type = EncoderType.ENVIRONMENT_ONLY
        else:  # FULL_BLACKBOX
            encoder_type = EncoderType.POLICY_ONLY

        self._encoder_service = EncoderService(
            config,
            env=self.env,
            algorithm=self.algorithm,
            encoder_type=encoder_type
        )


    @property
    def encoder_service(self):
        return self._encoder_service


    @encoder_service.setter
    def encoder_service(self, value):
        self._encoder_service = value


    def get_privacy_mode(self):
        return self.privacy_mode


    def get_encoded_state(self) -> torch.Tensor:
        """Get encoded state representation based on privacy mode."""
        return self._encoder_service.encode_state(
            self.env.altitude,
            self.algorithm.transitions
        )


    def reset(self) -> Tuple[VictimEnvironment, Algorithm]:
        """Reset the victim environment and algorithm"""
        self.env.reset_altitude()
        self.algorithm.reset()
        return self.env, self.algorithm

    def train(self, num_episodes: int):
        """Train using encoded states"""
        for episode in range(num_episodes):
            state = self.get_encoded_state()
            done = False

            while not done:
                # Get action from policy using encoded state
                action = self.algorithm.act(state)

                # Execute action in environment
                next_state_raw, reward, done, _ = self.env.step(action)

                # Get encoded next state
                next_state = self._encoder_service.encode_state(
                    self.env.altitude,
                    self.algorithm.transitions
                )

                # Update policy using encoded states
                self.algorithm.update(state, action, reward, next_state, done)
                state = next_state

    def get_current_representation(self):
        """
        Provides current victim representation depending on privacy mode.
        """
        encoded_state = self.get_encoded_state()

        if self.privacy_mode == PrivacyMode.FULL_WHITEBOX:
            return {
                'policy': self.algorithm.get_policy(),
                'transitions': self.env.T.copy(),
                'encoded_state': encoded_state
            }
        elif self.privacy_mode == PrivacyMode.PARTIAL_BLACKBOX:
            trajectories = self.sample_behavior(episodes=10)
            partial_policy = self.estimate_partial_policy(trajectories)
            return {
                'partial_policy': partial_policy,
                'trajectories': trajectories,
                'encoded_state': encoded_state
            }
        else:  # FULL_BLACKBOX
            return {
                'encoded_state': encoded_state
            }

    def sample_behavior(self, episodes: int):
        # Implement sampling victim episodes and recording trajectories
        pass


    def estimate_partial_policy(self, trajectories):
        # Implement estimating partial policy (Q-values distribution) from trajectories
        pass
