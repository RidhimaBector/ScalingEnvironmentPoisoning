

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

from agent.attack_dispatch import AttackDispatch
from agent.victim_system import VictimSystem
from envs.target_def import create_target_policy
from utils.utils_attack import Attack_Done_Identify


class AttackEnvironment(Env):
    """Gymnasium environment for the environment poisoning attack MDP.

    Observation: raw env_dynamics vector (shape = nS).
    Action: dispatchable perturbation vector (shape = nS) in [-1, 1].
    Reward: mean victim accuracy vs target policy.
    Done: when all victims converge (min accuracy == 1.0).

    Args:
        victim_population: VictimSystem managing K victims.
        attack_dispatch: AttackDispatch for applying perturbations.
        target_policy: Target policy matrix (nS, nA) or None for default.
        victim_train_episodes: Episodes to train victims per attack step.
        config: Optional config object (unused here, kept for call-site compat).
        target_path_type: Path type for default target policy construction.
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        victim_population: VictimSystem,
        attack_dispatch: Optional[AttackDispatch] = None,
        target_policy: Optional[np.ndarray] = None,
        victim_train_episodes: int = 80,
        config: Any = None,
        target_path_type: str = "Mp",
    ):
        super().__init__()

        self._victim_population = victim_population
        self._dispatch = attack_dispatch or AttackDispatch()
        self._victim_train_episodes = victim_train_episodes

        # Target policy
        if target_policy is not None:
            self._target = target_policy
        else:
            self._target = create_target_policy(
                victim_population.nS, victim_population.nA, path_type=target_path_type
            )

        env_nS = victim_population.nS

        # Gym spaces — obs is raw env_dynamics (nS,); action is perturbation (nS,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(env_nS,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(env_nS,), dtype=np.float64
        )

        # Convenience attribute used by main.py to get the env dimension
        self.nS = env_nS
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset attack environment for a new episode.

        Returns:
            Tuple (env_dynamics, info).
        """
        if seed is not None:
            self._victim_population.seed(seed)

        self._step_count = 0
        self._victim_population.reset()

        obs = self._get_env_dynamics()
        return obs, {'step': 0, 'is_initial': True}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one attack step.

        1. Apply perturbation to all victim environments via dispatch.
        2. Train all victims for victim_train_episodes.
        3. Compute reward from accuracy metrics.

        Args:
            action: Dispatchable perturbation vector of shape (nS,).
                The ActionTranslator in the training loop is responsible
                for clipping/scaling before passing here.

        Returns:
            Tuple (obs, reward, terminated, truncated, info) where obs is
            the raw env_dynamics array.  info['victim_results'] contains
            the full per-victim result dicts.
        """
        self._step_count += 1

        # 1. Dispatch perturbation
        self._dispatch.apply(self._victim_population, action)

        # 2. Train all victims (logs to Sacred tracker if attached)
        results = self._victim_population.run_experiments(self._victim_train_episodes)

        # 3. Metrics and reward
        metrics = self._victim_population.get_accuracy_metrics(self._target)
        reward = float(metrics['mean_accuracy'])
        terminated = metrics['min_accuracy'] >= 1.0

        obs = self._get_env_dynamics()

        info = {
            'step': self._step_count,
            'accuracy': metrics['mean_accuracy'],
            'min_accuracy': metrics['min_accuracy'],
            'max_accuracy': metrics['max_accuracy'],
            'std_accuracy': metrics['std_accuracy'],
            'mean_accuracy_sftmx': metrics.get('mean_accuracy_sftmx', 0.0),
            'per_victim_accuracy': metrics.get('per_victim_accuracy', []),
            'victim_results': results,
        }

        return obs, reward, terminated, False, info

    def render(self, mode='human'):
        if hasattr(self._victim_population.env, 'render_altitude_heatmap'):
            return self._victim_population.env.render_altitude_heatmap()

    def seed(self, seed=None):
        if seed is not None:
            self._victim_population.seed(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_env_dynamics(self) -> np.ndarray:
        return self._victim_population.get_env_dynamics().flatten()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def victim_system(self) -> VictimSystem:
        return self._victim_population

    @victim_system.setter
    def victim_system(self, value: VictimSystem):
        self._victim_population = value

    @property
    def num_victims(self) -> int:
        return self._victim_population.num_victims

    def get_accuracy_metrics(self) -> Dict[str, float]:
        return self._victim_population.get_accuracy_metrics(self._target)

    def is_done(self, threshold: float = 1.0) -> bool:
        return self._victim_population.is_all_converged(threshold)
