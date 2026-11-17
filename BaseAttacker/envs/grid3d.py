"""Grid world with altitude: the main environment of the paper.

A rows x cols grid. The victim starts in the top-left cell and must reach the
bottom-left cell; every step costs 1. Each cell has an altitude in [0, 10], and
altitude differences decide how a move plays out:

  uphill    the victim often fails to move (up to 90% of the time on a steep climb)
  downhill  the victim may slide one cell further (up to 20% of the time)
  flat      the move usually succeeds

Naturally the altitude falls from the top row (8) to the bottom row (2), so heading
straight down is cheapest. The attacker changes altitudes by up to 1 per cell per
attack step. Its target is the perimeter route: along the top row, down the right
column, back along the bottom row.
"""

import math
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from core.victim_environment import TabularModel, VictimEnvironment
from envs.target_def import create_target_policy
from utils.metrics import Attack_Cost_Compute_K, Attack_Cost_Compute_W, greedy_path

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
ALTITUDE_MIN, ALTITUDE_MAX = 0.0, 10.0
STRUGGLE = 0.9  # base probability of failing an uphill move
SLIDE = 0.2  # base probability of sliding an extra cell downhill


class Grid3D(VictimEnvironment):
    """See the module docstring. `grid_dimensions` is (rows, cols)."""

    def __init__(self, grid_dimensions: Tuple[int, int] = (4, 4)):
        super().__init__()
        self.shape = tuple(int(v) for v in grid_dimensions)
        self.nS = int(np.prod(self.shape))
        self.nA = 4
        self.max_steps = 1000  # per victim episode
        self.start_state = 0
        self._goal_state = int(np.ravel_multi_index((self.shape[0] - 1, 0), self.shape))

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.altitude_default = self._natural_altitude()
        self.altitude = self.altitude_default.copy()
        self.T = self._transitions(self.altitude)
        self.INIT_T = self.T.copy()  # transitions of the unpoisoned grid
        self._prev_dynamics = None
        self.seed()
        self.s = self.start_state

    def __str__(self) -> str:
        return f"Grid3D(shape={self.shape})"

    # --- the game ---

    def _natural_altitude(self) -> np.ndarray:
        """Top row 8, bottom row 2, constant along each row."""
        rows, _ = self.shape
        altitude = np.zeros(self.shape)
        for r in range(rows):
            altitude[r, :] = round(8.0 - 6.0 * r / max(rows - 1, 1), 2)
        return altitude

    def _transitions(self, altitude: np.ndarray) -> np.ndarray:
        """Transition tensor T[action][state][next_state] for the given altitude map."""
        rows, cols = self.shape
        moves = (
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        )  # NORTH, EAST, SOUTH, WEST as (d_row, d_col)
        T = np.zeros((self.nA, self.nS, self.nS))
        for a in range(self.nA):
            T[a] = np.eye(self.nS)  # walls: stay in place

        for row in range(rows):
            for col in range(cols):
                s = row * cols + col
                for a, (dr, dc) in enumerate(moves):
                    r2, c2 = row + dr, col + dc
                    if not (0 <= r2 < rows and 0 <= c2 < cols):
                        continue
                    r3, c3 = row + 2 * dr, col + 2 * dc
                    can_slide = 0 <= r3 < rows and 0 <= c3 < cols

                    climb = altitude[r2][c2] - altitude[row][col]
                    slide = SLIDE / (1 + math.exp(2 + 2 * climb)) if can_slide else 0.0
                    stay = STRUGGLE / (1 + math.exp(2 - 2 * climb))
                    T[a][s][s] = stay
                    T[a][s][r2 * cols + c2] = 1 - slide - stay
                    if can_slide:
                        T[a][s][r3 * cols + c3] = slide

        eps = 1e-6  # keep every transition possible
        return (T + eps) / (1 + self.nS * eps)

    @property
    def T(self) -> np.ndarray:
        return self._T

    @T.setter
    def T(self, value: np.ndarray) -> None:
        self._T = value
        self._T_cdf = np.cumsum(value, axis=2)  # cache for fast sampling in step()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.s = self.start_state
        return self.s, {}

    def step(self, action: int):
        cdf = self._T_cdf[action][self.s]
        next_state = int(
            min(
                np.searchsorted(cdf, np.random.random() * cdf[-1], side="right"),
                self.nS - 1,
            )
        )
        prob = self._T[action][self.s][next_state]
        self.s = next_state
        return next_state, -1.0, next_state == self._goal_state, False, {"prob": prob}

    def tabular_model(self) -> TabularModel:
        terminal = np.zeros(self.nS, dtype=bool)
        terminal[self._goal_state] = True
        return TabularModel(
            self._T_cdf, self.start_state, terminal, np.full(self.nS, -1.0), self.max_steps
        )

    # --- VictimEnvironment interface: what the attacker can change ---

    @property
    def perturbation_space(self) -> spaces.Space:
        """One altitude change per cell, each in [-1, 1]."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.nS,), dtype=np.float64)

    def apply_perturbation(self, perturbation: np.ndarray) -> None:
        self._prev_dynamics = self.get_dynamics()
        self.altitude = np.clip(
            self.altitude + np.asarray(perturbation).reshape(self.shape),
            ALTITUDE_MIN,
            ALTITUDE_MAX,
        )
        self.T = self._transitions(self.altitude)

    def get_dynamics(self) -> np.ndarray:
        return self.altitude.copy().flatten()

    def reset_dynamics(self) -> None:
        self.altitude = self.altitude_default.copy()
        self.T = self._transitions(self.altitude)

    @property
    def dynamics_dim(self) -> int:
        return self.nS

    def dynamics_bounds(self):
        return (ALTITUDE_MIN, ALTITUDE_MAX)

    # --- the attacker's goal, and ground truth for evaluation ---

    def make_target_policy(self, path_type: str = "Mp") -> np.ndarray:
        return create_target_policy(
            self.nS, self.nA, path_type=path_type, shape=self.shape
        )

    def oracle_route_match(self, target_actions, greedy_actions) -> float:
        want = greedy_path(
            self.INIT_T, target_actions, self.start_state, self._goal_state
        )
        got = greedy_path(self.T, greedy_actions, self.start_state, self._goal_state)
        return float(got == want)

    def compute_distance_metrics(self, algo, target: np.ndarray) -> Dict[str, float]:
        """White-box only: Wasserstein and KL distance rates between the victim's policy
        and the target, negated so that higher means closer."""
        try:
            import ot  # optional dependency (POT), only needed here
        except ImportError as exc:
            raise ImportError("privacy_mode=full_whitebox on the grid needs the POT package: "
                              "pip install POT") from exc
        coords = np.array(
            [[i, j] for i in range(self.shape[0]) for j in range(self.shape[1])]
        )
        M = ot.dist(coords, coords, metric="cityblock") * 20
        np.fill_diagonal(M, 10)
        M = np.repeat(np.repeat(M, self.nA, axis=1), self.nA, axis=0)
        np.fill_diagonal(M, 0)
        Q = algo.get_policy_matrix()
        out = {}
        for name, kind in (("", 0), ("_grid", 1), ("_behavior", 2)):
            out[f"distance{name}_K"] = -Attack_Cost_Compute_K(
                self, self.INIT_T, Q, target, M, distance_type=kind
            )
            out[f"distance{name}_W"] = -Attack_Cost_Compute_W(
                self, self.INIT_T, Q, target, M, distance_type=kind
            )
        return out
