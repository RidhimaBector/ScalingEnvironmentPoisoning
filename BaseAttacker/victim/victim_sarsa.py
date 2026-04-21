"""SARSA victim algorithm.

Implements the VictimAlgorithm ABC. On-policy variant of Q-learning:
the TD update uses the actual next action chosen by the policy rather
than the greedy max action.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
from core.victim_algorithm import VictimAlgorithm
from scipy.special import softmax

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.victim_environment import VictimEnvironment


def _get_max_steps(env, default: int = 1000) -> int:
    """Return the episode step limit from env.max_steps or env.spec."""
    if hasattr(env, 'max_steps') and env.max_steps is not None:
        return int(env.max_steps)
    spec = getattr(env, 'spec', None)
    if spec is not None and getattr(spec, 'max_episode_steps', None) is not None:
        return int(spec.max_episode_steps)
    return default


class VictimSARSA(VictimAlgorithm):
    """SARSA victim algorithm.

    Maintains a Q-table updated via on-policy TD learning.
    Drop-in replacement for VictimQLearning via algo_class=VictimSARSA.
    """

    def __init__(self, env_nS: int, env_nA: int, memory_size: int,
                 discount_factor: float = 1.0, alpha: float = 0.1,
                 epsilon: float = 0.1):
        self.env_nS = env_nS
        self.env_nA = env_nA
        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self._init_structures()

    def _init_structures(self) -> None:
        self.Q = np.zeros((self.env_nS, self.env_nA))
        self._transitions = np.ones((self.env_nS, 2)) * -1
        self._transitions[:, 0] = np.arange(self.env_nS)

    def act(self, state: int) -> int:
        action_probs = softmax(self.Q[state])
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def reset(self) -> None:
        self._init_structures()

    def train(self, env: VictimEnvironment, num_episodes: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            'episode_rewards': [],
            'episode_lengths': [],
            'transitions': self._transitions.copy(),
        }

        for _ in range(num_episodes):
            episode_stats = self._run_episode(env)
            stats['episode_rewards'].append(episode_stats['reward'])
            stats['episode_lengths'].append(episode_stats['length'])
            stats['transitions'] = episode_stats['transitions']

        return stats

    def _run_episode(self, env: VictimEnvironment) -> Dict[str, Any]:
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        action = self.act(state)
        episode_reward = 0.0
        transitions = []

        max_steps = _get_max_steps(env)
        for t in range(max_steps):
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result if len(result) == 5 else (*result[:3], False, result[3])
            done = terminated or truncated
            self._transitions[state, 1] = action
            transitions.append((state, action))

            next_action = self.act(next_state)

            # SARSA update: use actual next_action (on-policy)
            td_target = reward + self.discount_factor * self.Q[next_state][next_action]
            td_delta = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_delta

            episode_reward += reward
            if done:
                break
            state = next_state
            action = next_action

        return {
            'reward': episode_reward,
            'length': t + 1,
            'transitions': transitions,
        }

    def get_policy_matrix(self) -> np.ndarray:
        return self.Q

    def get_behavior_trace(self) -> np.ndarray:
        return self._transitions.copy()

    def save(self, path: str) -> None:
        np.save(f"{path}_q_table.npy", self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(f"{path}_q_table.npy")
