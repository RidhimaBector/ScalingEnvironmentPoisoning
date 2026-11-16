"""Name -> class registries for victim environments and victim algorithms.

Adding a new game or a new victim learner means implementing the interface in
core/ (VictimEnvironment or VictimAlgorithm) and adding one line here. Nothing
else in the attack pipeline refers to a concrete environment or algorithm.
"""
from envs.chain import SlipperyRing
from envs.env3D_4x4 import Grid3D
from victim.victim_Q import VictimQLearning
from victim.victim_reinforce import VictimREINFORCE
from victim.victim_sarsa import VictimSARSA

ENV_MAP = {
    "grid3d": Grid3D,
    "ring": SlipperyRing,
}

VICTIM_ALGO_MAP = {
    "qlearning": VictimQLearning,
    "sarsa": VictimSARSA,
    "reinforce": VictimREINFORCE,
}
