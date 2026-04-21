"""Core abstract interfaces for the environment poisoning attack framework.

Defines swappable ABCs for all major components:
- VictimAlgorithm: RL algorithm being attacked
- VictimEnvironment: Environment the victim interacts with
- Encoder: Encodes victim data into attack state embeddings
- AttackAlgorithm: RL algorithm that learns the attack policy

These interfaces allow different implementations (Q-learning, DQN, PPO for victims;
Grid3D, LunarLander for environments; etc.) to be used interchangeably.
"""

from core.types import PrivacyMode
from core.victim_algorithm import VictimAlgorithm
from core.victim_environment import VictimEnvironment
from core.encoder import Encoder
from core.attack_algorithm import AttackAlgorithm

__all__ = [
    "PrivacyMode",
    "VictimAlgorithm",
    "VictimEnvironment",
    "Encoder",
    "AttackAlgorithm",
]
