"""Name -> class registries for victim environments and victim algorithms.

Adding a new game or a new victim learner means implementing the interface in
core/ (VictimEnvironment or VictimAlgorithm) and adding one line here. Nothing
else in the attack pipeline refers to a concrete environment or algorithm.

Environments are imported on first use, so an optional dependency (Box2D for the
lander) is only needed when that environment is actually selected.
"""
import importlib

from victim.victim_dqn import VictimDQN
from victim.victim_Q import VictimQLearning
from victim.victim_reinforce import VictimREINFORCE
from victim.victim_sarsa import VictimSARSA


class _LazyClasses(dict):
    """name -> 'module:Class', resolved to the class when looked up."""

    def __getitem__(self, name):
        module, cls = super().__getitem__(name).split(":")
        return getattr(importlib.import_module(module), cls)

    def get(self, name, default=None):
        return self[name] if name in self else default


ENV_MAP = _LazyClasses({
    "grid3d": "envs.env3D_4x4:Grid3D",
    "ring": "envs.chain:SlipperyRing",
    "lunarlander": "envs.lunar_lander:LunarLanderAttack",
})

VICTIM_ALGO_MAP = {
    "qlearning": VictimQLearning,
    "sarsa": VictimSARSA,
    "reinforce": VictimREINFORCE,
    "dqn": VictimDQN,
}
