"""A second, deliberately different victim environment: a slippery ring.

It exists to show the attack pipeline does not depend on the grid. It differs from
Grid3D in every way the pipeline could have hard-coded: 2 actions instead of 4, no
2-D shape, a perturbation vector whose meaning is friction rather than altitude,
different bounds, and its own target policy.

N states on a ring. The victim starts at state 0 and must reach state N // 2.
Action 0 steps clockwise (+1), action 1 steps counter-clockwise (-1). Leaving state
s succeeds with probability 1 - 0.9 * friction[s]; otherwise the victim stays put.
Reward is -1 per step. Naturally the clockwise half is smooth and the other half is
sticky, so the victim goes clockwise. The attacker's target is the counter-clockwise
route.
"""
from typing import Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from core.victim_environment import TabularModel, VictimEnvironment


class SlipperyRing(VictimEnvironment):
    STEP = 0.1  # friction change per unit of perturbation

    def __init__(self, n_states: int = 8):
        super().__init__()
        self.nS = int(n_states)
        self.nA = 2
        self.goal = self.nS // 2
        self.max_steps = 200
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        # counter-clockwise half (states goal+1 .. N-1) is sticky
        self.friction_default = np.full(self.nS, 0.1)
        self.friction_default[self.goal + 1:] = 0.6
        self.friction = self.friction_default.copy()
        self._prev_dynamics = None
        self.s = 0

    # --- gym API ---
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.s = 0
        return self.s, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        if np.random.random() < 1.0 - 0.9 * self.friction[self.s]:
            self.s = (self.s + (1 if action == 0 else -1)) % self.nS
        return self.s, -1.0, self.s == self.goal, False, {}

    def tabular_model(self) -> TabularModel:
        move = 1.0 - 0.9 * self.friction  # chance that a move out of each state succeeds
        T = np.zeros((self.nA, self.nS, self.nS))
        states = np.arange(self.nS)
        for action, step in ((0, 1), (1, -1)):
            T[action, states, states] = 1.0 - move
            T[action, states, (states + step) % self.nS] += move
        terminal = np.zeros(self.nS, dtype=bool)
        terminal[self.goal] = True
        return TabularModel(
            np.cumsum(T, axis=2), 0, terminal, np.full(self.nS, -1.0), self.max_steps
        )

    # --- VictimEnvironment interface ---
    @property
    def perturbation_space(self) -> spaces.Space:
        return spaces.Box(low=-1.0, high=1.0, shape=(self.nS,), dtype=np.float64)

    def apply_perturbation(self, perturbation: np.ndarray) -> None:
        self._prev_dynamics = self.get_dynamics().copy()
        self.friction = np.clip(self.friction + self.STEP * np.asarray(perturbation).flatten(), 0.0, 1.0)

    def get_dynamics(self) -> np.ndarray:
        return self.friction.copy()

    def reset_dynamics(self) -> None:
        self.friction = self.friction_default.copy()

    @property
    def dynamics_dim(self) -> int:
        return self.nS

    def dynamics_bounds(self):
        return (0.0, 1.0)

    def make_target_policy(self, path_type: str = "Mp") -> np.ndarray:
        """Counter-clockwise from the start: action 1 in states 0, N-1, ..., goal+1."""
        target = np.zeros((self.nS, self.nA))
        for s in [0] + list(range(self.goal + 1, self.nS)):
            target[s, 1] = 1.0
        return target

    def oracle_route_match(self, target_actions, greedy_actions):
        states = [0] + list(range(self.goal + 1, self.nS))
        return float(all(int(greedy_actions[s]) == 1 for s in states))
