"""Q-learning victim algorithm.

Implements the VictimAlgorithm ABC. Maintains a Q-table updated via
TD learning and tracks behavior traces for blackbox/whitebox encoding.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import utils.utils_buf as utils_buf
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


class VictimQLearning(VictimAlgorithm):
    """Q-learning victim algorithm.

    Maintains a Q-table updated via TD learning. Tracks behavior traces
    (last action taken per state) for blackbox encoding.
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
        self.MEM = utils_buf.Memory(self.memory_size)
        self._transitions = np.ones((self.env_nS, 2)) * -1
        self._transitions[:, 0] = np.arange(self.env_nS)
        self._trajectory_buffer: list = []

    @property
    def transitions(self) -> np.ndarray:
        return self._transitions

    def act(self, state: int) -> int:
        action_probs = softmax(self.Q[state])
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta

    def _run_episode(self, env: VictimEnvironment) -> Dict[str, Any]:
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        episode_reward = 0
        transitions = []
        prev_state = None
        prev_action = None

        max_steps = _get_max_steps(env)
        for t in range(max_steps):
            action = self.act(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result if len(result) == 5 else (*result[:3], False, result[3])
            done = terminated or truncated

            self._transitions[state, 1] = action
            self.update(state, action, reward, next_state, done)
            transitions.append((state, action))
            episode_reward += reward

            if prev_state is not None:
                self._trajectory_buffer.append((prev_state, prev_action, state, action))

            if done:
                break
            prev_state = state
            prev_action = action
            state = next_state

        return {
            'reward': episode_reward,
            'length': t + 1,
            'transitions': transitions,
        }

    def get_trajectories(self) -> list:
        out = list(self._trajectory_buffer)
        self._trajectory_buffer.clear()
        return out

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

    def reset(self) -> None:
        self._init_structures()
        self._trajectory_buffer = []

    def save(self, path: str) -> None:
        np.save(f"{path}_q_table.npy", self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(f"{path}_q_table.npy")

    # --- VictimAlgorithm ABC ---

    def get_policy_matrix(self) -> np.ndarray:
        return self.Q

    def get_behavior_trace(self) -> np.ndarray:
        return self._transitions.copy()

    def get_greedy_actions(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
