from algorithms.algorithm import Algorithm
from envs.environment import Environment
import torch
from ae.ae import AutoEncoder
from .autoencoder import EnvAutoEncoder
import numpy as np
from typing import TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from agent.victim_system import VictimSystem


class EncoderType(Enum):
    POLICY_ONLY = "policy"
    ENVIRONMENT_ONLY = "environment"
    COMBINED = "combined"
    NIL = "nil"
    WHITEBOX = "whitebox"


class EncoderService:
    def __init__(self, config, env: Environment = None, algorithm: Algorithm = None, encoder_type: EncoderType = EncoderType.COMBINED):
        self.embedding_len = config.AE.EMBEDDING_SIZE
        self.encoder_type = encoder_type

        if self.encoder_type == EncoderType.NIL or self.encoder_type == EncoderType.WHITEBOX:
            self.policy_encoder = None
            self.env_encoder = None
        elif self.encoder_type == EncoderType.POLICY_ONLY:
            self._setup_policy_encoder(algorithm)
            self.env_encoder = None
        elif self.encoder_type == EncoderType.ENVIRONMENT_ONLY:
            self._setup_env_encoder(env, config)
            self.policy_encoder = None
        elif self.encoder_type == EncoderType.COMBINED:
            self._setup_policy_encoder(algorithm)
            self._setup_env_encoder(env, config)


    def _setup_policy_encoder(self, algorithm: Algorithm):
        policy_kwargs = {
                "enc_in_size": 32,
                "enc_out_size": 5,
                "dec_in_size": 6,
                "dec_out_size": 5,
                "lr": 0.001
            }
        self.policy_encoder = AutoEncoder(**policy_kwargs)
        self.policy_encoder.load("/Users/kunwarnir/projects/envPoisoning/master/ScalingEnvironmentPoisoning/BaseAttacker/ae/models/340240_f-o_AutoEncoder_SftMx")


    def _setup_env_encoder(self, env: Environment, config):
        # Environment encoder setup using ENV_KWARGS
        env_kwargs = {
            "env": env,
            "enc_in_size": config.AE.ENV_KWARGS.enc_in_size,
            "enc_out_size": config.AE.ENV_KWARGS.enc_out_size,
            "enc_num_layer": config.AE.ENV_KWARGS.enc_num_layer,
            "dec_fc_in_size": config.AE.ENV_KWARGS.dec_fc_in_size,
            "dec_fc_out_size": env.nA,  # number of actions
            "dec_lstm_in_size": config.AE.ENV_KWARGS.dec_lstm_in_size,
            "dec_lstm_out_size": env.nS,  # number of states
            "dec_lstm_num_layer": config.AE.ENV_KWARGS.dec_lstm_num_layer,
            "seq_len": config.AE.SEQ_LEN,
            "embedding_len": config.AE.EMBEDDING_SIZE,
            "n_epochs": config.AE.ENV_KWARGS.n_epochs,
            "lr": config.AE.ENV_KWARGS.lr
        }
        self.env_encoder = EnvAutoEncoder(**env_kwargs)


    def encode(self, victim_system: 'VictimSystem'):
        # if self.encoder_type == EncoderType.POLICY_ONLY:
        #     policy_embedding = self.policy_encoder.Policy_Embedding(policy_transitions)
        #     return torch.from_numpy(policy_embedding[-1]).unsqueeze(0)
        # elif self.encoder_type == EncoderType.ENVIRONMENT_ONLY:
        #     return self.env_encoder.encode_environment(env_state).view(1, -1)
        # else:  # COMBINED - original behavior
        #     policy_embedding = self.policy_encoder.Policy_Embedding(policy_transitions)
        #     env_embedding = self.env_encoder.encode_environment(env_state)
        #     return torch.cat((torch.from_numpy(policy_embedding[-1]).unsqueeze(0), env_embedding.view(1, -1)), dim=3)

        # Handle whitebox encoding first
        if self.encoder_type == EncoderType.WHITEBOX:
            return self._encode_whitebox(victim_system)

        # Original encoding logic (commented out above, reimplemented below)
        if self.policy_encoder is None and self.env_encoder is None:
            return torch.zeros((1, self.embedding_len)).unsqueeze(0).unsqueeze(0)
        elif self.policy_encoder is None:
            policy_encoding = torch.zeros((1, self.embedding_len)).unsqueeze(0).unsqueeze(0)
            env_encoding = self.env_encoder.encode_environment(victim_system.env.altitude)
            return torch.cat((policy_encoding, env_encoding), dim=3)
        elif self.env_encoder is None:
            policy_encoding = self.policy_encoder.Policy_Embedding(victim_system.algorithm.transitions)
            env_encoding = torch.zeros((1, victim_system.env.nS)).unsqueeze(0).unsqueeze(0)
            return torch.cat((policy_encoding, env_encoding), dim=3)
        else:  # COMBINED - original behavior
            policy_encoding = self.policy_encoder.Policy_Embedding(victim_system.algorithm.transitions)
            env_encoding = self.env_encoder.encode_environment(victim_system.env.altitude)
            return torch.cat((policy_encoding, env_encoding), dim=3)

    def _encode_whitebox(self, victim_system: 'VictimSystem') -> torch.Tensor:
        """Whitebox encoding that returns raw victim info and environment state."""
        # Get victim policy information (flattened Q-values)
        victim_info = victim_system.algorithm.Q.flatten()
        victim_tensor = torch.from_numpy(victim_info).float()
        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Get environment state (altitude)
        env_info = victim_system.env.altitude.copy()
        env_tensor = torch.from_numpy(env_info).float()
        env_tensor = env_tensor.view(1, victim_system.env.nS)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        # Concatenate victim and environment information
        encoding = torch.cat((victim_tensor_4d, env_tensor_4d), dim=3)
        return encoding

    def _gather_victim_data(self, victim_env, victim_algo):
        # Victim data can include:
        # - policy derived from Q-values
        # - environment dynamics ("T")
        # Implement carefully here depending on encoder training
        policy = get_policy_from_Q(victim_algo.Q)
        env_dynamics = victim_env.T
        return np.concatenate([policy.flatten(), env_dynamics.flatten()])

    def get_initial_state(self, victim_system: 'VictimSystem') -> torch.Tensor:
        if self.encoder_type == EncoderType.WHITEBOX:
            # For whitebox, return zeros with appropriate dimensions
            victim_info = torch.zeros((1, victim_system.env.nS * victim_system.env.nA))
            victim_tensor_4d = victim_info.unsqueeze(0).unsqueeze(0)
            env_info = torch.zeros((1, victim_system.env.nS))
            env_tensor_4d = env_info.unsqueeze(0).unsqueeze(0)
            encoding = torch.cat((victim_tensor_4d, env_tensor_4d), dim=3)
            return encoding
        elif self.encoder_type == EncoderType.POLICY_ONLY:
            victim_info = torch.zeros((1, self.embedding_len))
            return victim_info.unsqueeze(0).unsqueeze(0)
        elif self.encoder_type == EncoderType.ENVIRONMENT_ONLY:
            env_info = torch.zeros((1, victim_system.env.nS))
            return env_info.unsqueeze(0).unsqueeze(0)
        else:  # COMBINED - original behavior
            victim_info = torch.zeros((1, self.embedding_len))
            victim_tensor_4d = victim_info.unsqueeze(0).unsqueeze(0)
            env_info = torch.zeros((1, victim_system.env.nS))
            env_tensor_4d = env_info.unsqueeze(0).unsqueeze(0)
            encoding = torch.cat((victim_tensor_4d, env_tensor_4d), dim=3)
            return encoding

    def encode_state(self, env_state: np.ndarray, policy_transitions: np.ndarray) -> torch.Tensor:
        if self.encoder_type == EncoderType.WHITEBOX:
            return torch.from_numpy(env_state).float()
        return self.env_encoder.encode_environment(env_state)
