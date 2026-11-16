

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

from agent.attack_dispatch import AttackDispatch
from agent.victim_system import VictimSystem
from envs.target_def import create_target_policy
from utils.utils_attack import Attack_Done_Identify, accuracy_from_trajectories
from core.encoder import Encoder


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
        encoder: Optional[Encoder] = None,
        reward_mode: str = "trace",
        obs_dynamics: str = "env",
        effort_weight: float = 0.0,
    ):
        super().__init__()

        self._victim_population = victim_population
        self._dispatch = attack_dispatch or AttackDispatch()
        self._victim_train_episodes = victim_train_episodes
        self._encoder = encoder
        self._reward_mode = reward_mode
        # What the attacker is shown next to the victim-behaviour embedding:
        #   "actions": the running sum of the perturbations it has sent (no read access
        #              to the environment; works for any environment)
        #   "env":     the environment's own dynamics vector (legacy)
        #   "none":    nothing
        if obs_dynamics not in ("actions", "env", "none"):
            raise ValueError(f"obs_dynamics must be actions|env|none, got {obs_dynamics!r}")
        self._obs_dynamics = obs_dynamics
        # Weight of the attack-size penalty in reward_mode="embedding" (0.1 in the
        # original lander attack). Measured from the attacker's own actions.
        self._effort_weight = effort_weight

        # Target policy: explicit > environment's own > grid default
        if target_policy is not None:
            self._target = target_policy
        else:
            self._target = victim_population.target

        env_nS = victim_population.nS

        # Obs space: encoded embedding when encoder is injected; raw dynamics otherwise
        ref_env = victim_population.env
        if encoder is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(encoder.embedding_dim,), dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(ref_env.dynamics_dim,), dtype=np.float64
            )
        if hasattr(ref_env, 'perturbation_space'):
            self.action_space = ref_env.perturbation_space
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(env_nS,), dtype=np.float64
            )
        self._sent = np.zeros(self.action_space.shape)  # running sum of perturbations sent

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
        self._sent = np.zeros(self.action_space.shape)
        self._victim_population.reset()

        if self._encoder is not None:
            if hasattr(self._encoder, 'reset_episode'):
                self._encoder.reset_episode()
            obs = self._encoder.get_initial_embedding()
        else:
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
        self._sent = self._sent + np.clip(
            np.asarray(action, dtype=float).reshape(self._sent.shape),
            self.action_space.low, self.action_space.high)

        # 2. Train all victims (logs to Sacred tracker if attached)
        results = self._victim_population.run_experiments(self._victim_train_episodes)

        # 3. Metrics and reward
        if self._encoder is not None:
            obs = self._encoder.encode(results, self._attacker_context())
            if self._target is not None:
                trace_acc = self._encoder.compute_reward(results, self._target)
                traj_acc = float(np.mean([
                    accuracy_from_trajectories(r.get('trajectories', []), self._target)
                    for r in results
                ]))
            else:
                trace_acc = traj_acc = 0.0   # no per-state target table
            if self._reward_mode == "embedding":
                # Similarity of the victim's behaviour embedding to the target's, minus a
                # penalty on how much the attacker has changed (as in the lander attack).
                sim = self._encoder.target_similarity()
                span = np.maximum(np.abs(self.action_space.high), np.abs(self.action_space.low))
                effort = float(np.linalg.norm(self._sent) /
                               max(np.linalg.norm(span * max(self._step_count, 1)), 1e-8))
                reward = (1 - self._effort_weight) * sim - self._effort_weight * effort
            elif self._reward_mode == "traj":
                reward = traj_acc
            elif self._reward_mode == "mix":
                reward = 0.5 * (trace_acc + traj_acc)
            else:
                reward = trace_acc
            # Never terminate on the noisy observed signal: run the full horizon.
            terminated = False
            info = {
                'step': self._step_count,
                'accuracy': reward,
                'trace_accuracy': trace_acc,
                'traj_accuracy': traj_acc,
                'target_similarity': self._encoder.target_similarity()
                if self._reward_mode == "embedding" else 0.0,
                'victim_results': results,
            }
        else:
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

        # Oracle diagnostics (evaluation only; never fed to the attacker).
        info.update(self._oracle_metrics())

        # Surface per-victim scalar metrics into info so log_timestep() picks them up.
        # Single victim: flat keys (distance_K, effort, ...).
        # Multi-victim: namespaced keys (victim_0.distance_K, ...).
        multi = len(results) > 1
        for res in results:
            k = res.get('victim_idx', 0)
            prefix = f'victim_{k}.' if multi else ''
            for name, value in res.items():
                if isinstance(value, (int, float)) and name != 'victim_idx':
                    info[f'{prefix}{name}'] = value

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

    def _oracle_metrics(self) -> Dict[str, float]:
        """Ground-truth poisoning success from the victims' true greedy policies.

        greedy_accuracy: fraction of target states whose argmax-Q action is the target action.
        path_match: 1.0 if the greedy rollout from the start state visits exactly
            the target path, else 0.0 (mean over victims).
        """
        if self._target is None:
            return self._oracle_metrics_trajectory_target()
        accs, matches = [], []
        tgt_actions = np.argmax(self._target, axis=1)
        for env, algo in self._victim_population._victims:
            greedy = np.asarray(algo.get_greedy_actions())
            _, acc, _, _ = Attack_Done_Identify(self._target.copy(), algo.get_policy_matrix())
            accs.append(acc)
            match = env.oracle_route_match(tgt_actions, greedy) if hasattr(env, 'oracle_route_match') else None
            if match is not None:
                matches.append(float(match))
        return {'greedy_accuracy': float(np.mean(accs)),
                'path_match': float(np.mean(matches)) if matches else 0.0}

    def _oracle_metrics_trajectory_target(self) -> Dict[str, float]:
        """Ground truth when the target is recorded behaviour instead of a policy table:
        the share of target states in which the victim's greedy action equals the
        target's action. Evaluation only; never shown to the attacker."""
        env0 = self._victim_population.env
        if not hasattr(self, '_target_sa'):
            traj = env0.target_trajectories() if hasattr(env0, 'target_trajectories') else []
            self._target_sa = (np.stack([t[2] for t in traj]).astype(np.float32),
                               np.array([int(t[3]) for t in traj])) if traj else None
        if self._target_sa is None:
            return {'greedy_accuracy': 0.0, 'path_match': 0.0}
        states, actions = self._target_sa
        accs = [float(np.mean(algo.greedy_actions_for(states) == actions))
                for _env, algo in self._victim_population._victims
                if hasattr(algo, 'greedy_actions_for')]
        return {'greedy_accuracy': float(np.mean(accs)) if accs else 0.0, 'path_match': 0.0}

    def _attacker_context(self):
        """The non-behaviour part of the attacker's observation (see obs_dynamics)."""
        if self._obs_dynamics == "env":
            return self._get_env_dynamics()
        if self._obs_dynamics == "actions":
            return self._sent.flatten().copy()
        return None

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
