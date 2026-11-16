

from typing import Any, Dict, List, Optional, Type

import numpy as np

from core.types import PrivacyMode
from utils.metrics import accuracy_from_behavior_trace, policy_accuracy

# Re-export PrivacyMode for backward compatibility
__all__ = ["VictimSystem", "PrivacyMode"]


class VictimSystem:
    """Manages K independent victim agents for environment poisoning attacks.

    Each victim has its own environment and algorithm instance. All victim
    environments share the same dynamics, which the attacker modifies.

    The run_experiments() method trains all victims for N episodes, collects
    results respecting the privacy mode, and optionally logs to Sacred.

    Args:
        num_victims: Number of victim agents K (default 1).
        env_class: Class to instantiate victim environments.
        algo_class: Class to instantiate victim algorithms.
        config: YACS configuration object.
        privacy_mode: What the attacker can see about victims.
        base_seed: Base random seed (victim k gets seed base_seed + k).
        env_kwargs: Kwargs for environment constructor.
        algo_kwargs: Kwargs for algorithm constructor.
        victim_tracker: Optional VictimExperimentTracker for Sacred logging.
    """

    def __init__(
        self,
        env_class: Type,
        algo_class: Type,
        num_victims: int = 1,
        privacy_mode: PrivacyMode = PrivacyMode.FULL_BLACKBOX,
        base_seed: int = 0,
        env_kwargs: Optional[Dict] = None,
        algo_kwargs: Optional[Dict] = None,
        victim_tracker: Any = None,
        target_path_type: str = "Mp",
    ):
        self.num_victims = num_victims
        self.privacy_mode = privacy_mode
        self.base_seed = base_seed
        self.victim_tracker = victim_tracker

        # Store classes for creating victims
        self._env_class = env_class
        self._algo_class = algo_class
        self._env_kwargs = env_kwargs or {}
        self._algo_kwargs = algo_kwargs or {}

        # Victim pairs: list of (env, algo) tuples
        self._victims: List[tuple] = []

        self._create_victims()

        # Reference environment for shared properties
        self._ref_env = self._victims[0][0]

        # Behavior traces: (nS, 2) per victim; None when the environment has no state table
        self._behavior_traces = [
            self._init_behavior_trace(self.nS) for _ in range(self.num_victims)
        ]

        # The behaviour the attacker wants: a policy table (nS, nA) for tabular
        # environments, or None when the environment gives recorded trajectories instead.
        self._target = self._ref_env.make_target_policy(target_path_type)
        if self._target is None and self._ref_env.target_trajectories() is None:
            raise ValueError(
                f"{type(self._ref_env).__name__} defines no target: implement "
                "make_target_policy() or target_trajectories().")

    def _create_victims(self) -> None:
        """Create K independent (env, algo) pairs with different seeds."""
        for k in range(self.num_victims):
            seed = self.base_seed + k

            env = self._env_class(**self._env_kwargs)
            env.seed(seed)

            if hasattr(self._algo_class, 'from_env'):
                # Algorithms that size themselves from the environment (e.g. DQN)
                algo = self._algo_class.from_env(env, **self._algo_kwargs)
            else:
                env_nS = getattr(env, 'nS', None) or env.observation_space.n
                env_nA = getattr(env, 'nA', None) or env.action_space.n
                algo_kwargs = {
                    'env_nS': env_nS,
                    'env_nA': env_nA,
                    **self._algo_kwargs,
                }
                algo = self._algo_class(**algo_kwargs)

            self._victims.append((env, algo))

    def _init_behavior_trace(self, n_states):
        """Initialize behavior trace matrix (nS, 2), col1=-1 (unvisited). None if no state table."""
        if n_states is None:
            return None
        trace = np.ones((n_states, 2)) * -1
        trace[:, 0] = np.arange(n_states)
        return trace

    # --- Properties ---

    @property
    def nS(self) -> int:
        """Number of states in the environment."""
        n = getattr(self._ref_env, 'nS', None)
        if n is None:
            n = getattr(self._ref_env.observation_space, 'n', None)  # None for continuous states
        return None if n is None else int(n)

    @property
    def nA(self) -> int:
        """Number of actions in the environment."""
        n = getattr(self._ref_env, 'nA', None)
        return int(n if n is not None else self._ref_env.action_space.n)

    @property
    def target(self) -> np.ndarray:
        """Target policy matrix."""
        return self._target

    @property
    def env(self):
        """The first victim's environment (all copies share dynamics)."""
        return self._victims[0][0]

    @property
    def algorithm(self):
        """The first victim's learning algorithm."""
        return self._victims[0][1]

    @property
    def victims(self) -> List[tuple]:
        """All (environment, algorithm) pairs."""
        return self._victims

    # --- Core operations ---

    def apply_perturbation(self, perturbation: np.ndarray) -> None:
        """Apply perturbation to ALL victim environments.

        Args:
            perturbation: Perturbation vector matching env.perturbation_space.
        """
        for env, _algo in self._victims:
            env.apply_perturbation(perturbation)

    def run_experiments(self, num_episodes: int) -> List[Dict]:
        """Train each victim for num_episodes and collect results.

        Respects privacy mode: blackbox victims only expose behavior traces.
        Logs per-victim metrics to Sacred if victim_tracker is set.

        Args:
            num_episodes: Number of training episodes per victim.

        Returns:
            List of result dicts, one per victim.
        """
        results = []

        for k, (env, algo) in enumerate(self._victims):
            # Train the victim
            algo.train(env, num_episodes)

            # Update behavior trace
            if self.nS is None:
                pass  # continuous states: behaviour is exposed as trajectories only
            elif hasattr(algo, 'get_behavior_trace'):
                self._behavior_traces[k] = algo.get_behavior_trace()
            else:
                greedy_actions = algo.get_greedy_actions()
                for s in range(self.nS):
                    self._behavior_traces[k][s, 1] = greedy_actions[s]

            # Collect result based on privacy mode
            result = self._collect_victim_result(k, env, algo)
            results.append(result)

            # Log to Sacred if tracker available — use only observable data
            if self.victim_tracker is not None and self._target is None:
                # No target policy table: log the victim's own episode return instead
                stats = getattr(algo, 'last_stats', None) or {}
                rewards = stats.get('episode_rewards') or [0.0]
                self.victim_tracker.log_victim_step(
                    victim_idx=k, metrics={'episode_return': float(np.mean(rewards))})
            elif self.victim_tracker is not None:
                if self.privacy_mode == PrivacyMode.FULL_WHITEBOX:
                    policy = algo.get_policy_matrix()
                    _, acc, acc_sftmx, acc_sftmx_complete = policy_accuracy(
                        self._target.copy(), policy
                    )
                else:
                    acc = accuracy_from_behavior_trace(
                        self._behavior_traces[k], self._target
                    )
                    acc_sftmx = 0.0
                    acc_sftmx_complete = 0.0
                metrics = {'accuracy': acc}
                if self.privacy_mode == PrivacyMode.FULL_WHITEBOX:
                    metrics['accuracy_sftmx'] = acc_sftmx
                    metrics['accuracy_sftmx_complete'] = acc_sftmx_complete
                self.victim_tracker.log_victim_step(victim_idx=k, metrics=metrics)

        return results

    def _collect_victim_result(self, k: int, env, algo) -> Dict:
        """Collect result from one victim based on privacy mode.

        FULL_WHITEBOX writes Q-table + env dynamics + behavior trace to Sacred.
        FULL_BLACKBOX writes only the behavior trace.

        Args:
            k: Victim index.
            env: Victim's environment.
            algo: Victim's algorithm.

        Returns:
            Dict with victim data.
        """
        result = {
            'victim_idx': k,
            'trajectories': algo.get_trajectories(),
        }
        if self._behavior_traces[k] is not None:
            result['behavior_trace'] = self._behavior_traces[k].copy()
        stats = getattr(algo, 'last_stats', None)
        if stats and stats.get('episode_rewards'):
            result['victim_return'] = float(np.mean(stats['episode_rewards']))

        if self.privacy_mode == PrivacyMode.FULL_WHITEBOX:
            result.update(algo.get_whitebox_data())
            result.update(env.get_whitebox_data())
            result.update(env.compute_distance_metrics(algo, self._target))
            result['effort'] = env.compute_effort()

        return result

    def get_env_dynamics(self) -> np.ndarray:
        """Get current environment dynamics from reference environment.

        Returns:
            Dynamics array (for Grid3D: altitude reshaped to (nS, 1)).
        """
        return self._ref_env.get_dynamics()

    def get_accuracy_metrics(self, target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute accuracy statistics across all victims.

        Args:
            target: Target policy matrix (uses self._target if None).

        Returns:
            Dict with mean/min/max/std accuracy and per-victim lists.
        """
        target = target if target is not None else self._target
        accuracies = []
        accuracies_sftmx = []

        for _env, algo in self._victims:
            policy = algo.get_policy_matrix()
            done, acc, acc_sftmx, acc_sftmx_complete = policy_accuracy(
                target.copy(), policy
            )
            accuracies.append(acc)
            accuracies_sftmx.append(acc_sftmx)

        acc_arr = np.array(accuracies)
        acc_sftmx_arr = np.array(accuracies_sftmx)

        return {
            'mean_accuracy': float(np.mean(acc_arr)),
            'min_accuracy': float(np.min(acc_arr)),
            'max_accuracy': float(np.max(acc_arr)),
            'std_accuracy': float(np.std(acc_arr)),
            'mean_accuracy_sftmx': float(np.mean(acc_sftmx_arr)),
            'min_accuracy_sftmx': float(np.min(acc_sftmx_arr)),
            'max_accuracy_sftmx': float(np.max(acc_sftmx_arr)),
            'per_victim_accuracy': accuracies,
            'per_victim_accuracy_sftmx': accuracies_sftmx,
        }

    def get_behavior_traces(self) -> List[np.ndarray]:
        """Get behavior traces for all victims.

        Returns:
            List of (nS, 2) arrays, one per victim.
        """
        return [trace.copy() for trace in self._behavior_traces]

    def get_policy_matrices(self) -> List[np.ndarray]:
        """Get policy matrices for all victims.

        Returns:
            List of (nS, nA) arrays, one per victim.
        """
        return [algo.get_policy_matrix() for _env, algo in self._victims]

    def is_all_converged(self, threshold: float = 1.0) -> bool:
        """Check if all victims have converged to target policy."""
        metrics = self.get_accuracy_metrics()
        return metrics['min_accuracy'] >= threshold

    def reset(self) -> None:
        """Reset all victims and behavior traces."""
        for k, (env, algo) in enumerate(self._victims):
            env.reset_dynamics()
            algo.reset()
            self._behavior_traces[k] = self._init_behavior_trace(self.nS)

    def seed(self, seed: int) -> None:
        """Set random seed for all victims."""
        self.base_seed = seed
        for k, (env, _algo) in enumerate(self._victims):
            env.seed(seed + k)
