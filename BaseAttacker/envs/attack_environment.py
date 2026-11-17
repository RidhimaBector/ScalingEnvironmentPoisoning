"""The attacker's environment.

From the attacker's point of view, poisoning is itself a decision process:

    observation  an encoding of how the victims currently behave (+ a context vector)
    action       a perturbation of the victims' environment
    reward       how close the victims' observed behaviour is to the target

One step applies the perturbation to every victim's environment, lets every victim
relearn for a fixed number of episodes, and then encodes what they did.

This class knows nothing about a particular game or learner. It talks to the victims
through VictimSystem and to the environment through the VictimEnvironment interface.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

from agent.attack_dispatch import AttackDispatch
from agent.victim_system import VictimSystem
from utils.metrics import (
    accuracy_from_trajectories,
    route_prefix_from_trajectories,
    target_action_accuracy,
)

OBS_DYNAMICS = ("actions", "env", "none")
REWARD_MODES = ("trace", "traj", "mix", "route", "route_mix", "embedding")


class AttackEnvironment(Env):
    """Gymnasium environment for the poisoning attack.

    Args:
        victim_population: the victims (VictimSystem).
        encoder: ObservationEncoder turning victim results into the observation.
        attack_dispatch: applies a perturbation to the victims' environments.
        victim_train_episodes: victim training episodes per attack step.
        reward_mode: see defaults.py.
        obs_dynamics: what the attacker sees besides victim behaviour:
            "actions" the running sum of the perturbations it has sent,
            "env" the environment's dynamics vector, "none" nothing.
        effort_weight: weight of the attack-size penalty in reward_mode="embedding".
        hold_threshold: if set, the attacker stops perturbing while its own reward
            (observed behaviour only) is at or above this value: the step's perturbation
            is replaced by zeros, and the attack resumes if the reward falls below it.
    """

    def __init__(
        self,
        victim_population: VictimSystem,
        encoder,
        attack_dispatch: Optional[AttackDispatch] = None,
        victim_train_episodes: int = 80,
        reward_mode: str = "trace",
        obs_dynamics: str = "actions",
        effort_weight: float = 0.0,
        hold_threshold: Optional[float] = None,
    ):
        super().__init__()
        if obs_dynamics not in OBS_DYNAMICS:
            raise ValueError(
                f"obs_dynamics must be one of {OBS_DYNAMICS}, got {obs_dynamics!r}"
            )
        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}"
            )

        self._victims = victim_population
        self._encoder = encoder
        self._dispatch = attack_dispatch or AttackDispatch()
        self._victim_train_episodes = victim_train_episodes
        self._reward_mode = reward_mode
        self._obs_dynamics = obs_dynamics
        self._effort_weight = effort_weight
        self._hold_threshold = hold_threshold
        self._last_reward = None  # reward of the previous step of this episode
        self._target = victim_population.target  # policy table, or None
        self._target_states_actions = None  # lazily loaded recorded target

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(encoder.embedding_dim,), dtype=np.float64
        )
        self.action_space = victim_population.env.perturbation_space
        self._sent = np.zeros(
            self.action_space.shape
        )  # running sum of perturbations sent
        self._step_count = 0

    # --- gymnasium API ---

    def reset(self, *, seed=None, options=None):
        """Fresh victims, unperturbed environment."""
        if seed is not None:
            self._victims.seed(seed)
        self._step_count = 0
        self._last_reward = None
        self._sent = np.zeros(self.action_space.shape)
        self._victims.reset()
        self._encoder.reset_episode()
        return self._encoder.get_initial_embedding(), {"step": 0}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        held = (
            self._hold_threshold is not None
            and self._last_reward is not None
            and self._last_reward >= self._hold_threshold - 1e-9
        )
        if held:  # target behaviour observed: leave the environment as it is
            action = np.zeros(self.action_space.shape)
        self._dispatch.apply(self._victims, action)
        self._sent = self._sent + np.clip(
            np.asarray(action, dtype=float).reshape(self._sent.shape),
            self.action_space.low,
            self.action_space.high,
        )

        results = self._victims.run_experiments(self._victim_train_episodes)
        obs = self._encoder.encode(results, self._context())
        reward, parts = self._reward(results)
        self._last_reward = reward

        info = {
            "step": self._step_count,
            "accuracy": reward,
            "held": float(held),
            "victim_results": results,
            **parts,
        }
        # Ground truth for evaluation. Never part of the observation or the reward.
        info.update(self._oracle_metrics())
        # Per-victim scalars (e.g. the victim's own return) for logging
        many = len(results) > 1
        for res in results:
            prefix = f"victim_{res.get('victim_idx', 0)}." if many else ""
            for name, value in res.items():
                if isinstance(value, (int, float)) and name != "victim_idx":
                    info[f"{prefix}{name}"] = value
        return obs, reward, False, False, info

    # --- what the attacker is given ---

    def _context(self):
        if self._obs_dynamics == "env":
            return np.asarray(self._victims.env.get_dynamics(), dtype=float).flatten()
        if self._obs_dynamics == "actions":
            return self._sent.flatten().copy()
        return None

    def _reward(self, results) -> Tuple[float, Dict[str, float]]:
        """Reward from observed behaviour only, plus its components for logging."""
        trace = traj = route = similarity = 0.0
        if self._target is not None:
            trace = self._encoder.compute_reward(results, self._target)
            traj = float(
                np.mean(
                    [
                        accuracy_from_trajectories(
                            r.get("trajectories", []), self._target
                        )
                        for r in results
                    ]
                )
            )
            route = float(
                np.mean(
                    [
                        route_prefix_from_trajectories(
                            r.get("trajectories", []), self._target
                        )
                        for r in results
                    ]
                )
            )
        if self._reward_mode == "embedding":
            similarity = self._encoder.target_similarity()
            span = np.maximum(
                np.abs(self.action_space.high), np.abs(self.action_space.low)
            )
            effort = float(
                np.linalg.norm(self._sent)
                / max(np.linalg.norm(span * self._step_count), 1e-8)
            )
            reward = (
                1 - self._effort_weight
            ) * similarity - self._effort_weight * effort
        elif self._reward_mode == "route":
            reward = route
        elif self._reward_mode == "route_mix":
            reward = 0.5 * (trace + route)
        elif self._reward_mode == "traj":
            reward = traj
        elif self._reward_mode == "mix":
            reward = 0.5 * (trace + traj)
        else:
            reward = trace
        return float(reward), {
            "trace_accuracy": trace,
            "traj_accuracy": traj,
            "route_prefix": route,
            "target_similarity": similarity,
        }

    # --- ground truth (evaluation only) ---

    def _oracle_metrics(self) -> Dict[str, float]:
        """greedy_accuracy: share of target states where the victim's greedy action is
        the target's. path_match: 1 if the victim's greedy route is the target route
        (environments that define a route), averaged over victims."""
        accs, routes = [], []
        if self._target is not None:
            target_actions = np.argmax(self._target, axis=1)
            for env, algo in self._victims.victims:
                accs.append(
                    target_action_accuracy(self._target, algo.get_policy_matrix())
                )
                match = env.oracle_route_match(
                    target_actions, np.asarray(algo.get_greedy_actions())
                )
                if match is not None:
                    routes.append(float(match))
        else:
            if self._target_states_actions is None:
                recorded = self._victims.env.target_trajectories() or []
                self._target_states_actions = (
                    (
                        np.stack([t[2] for t in recorded]).astype(np.float32),
                        np.array([int(t[3]) for t in recorded]),
                    )
                    if recorded
                    else ()
                )
            if self._target_states_actions:
                states, actions = self._target_states_actions
                accs = [
                    float(np.mean(algo.greedy_actions_for(states) == actions))
                    for _env, algo in self._victims.victims
                ]
        return {
            "greedy_accuracy": float(np.mean(accs)) if accs else 0.0,
            "path_match": float(np.mean(routes)) if routes else 0.0,
        }

    # --- access ---

    @property
    def victim_system(self) -> VictimSystem:
        return self._victims

    @property
    def num_victims(self) -> int:
        return self._victims.num_victims
