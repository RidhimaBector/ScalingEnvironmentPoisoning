"""REINFORCE (Monte Carlo policy gradient) victim algorithm.

Implements the VictimAlgorithm ABC. Maintains a preference/logit matrix
H(nS, nA) and updates via the REINFORCE gradient with softmax policy.
"""

from __future__ import annotations

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


class VictimREINFORCE(VictimAlgorithm):
    """REINFORCE victim algorithm.

    Policy: pi(a|s) = softmax(H[s]).
    Update: H[s,a] += alpha * G_t * (I[a] - pi(s))  (score function gradient).
    """

    def __init__(self, env_nS: int, env_nA: int, memory_size: int,
                 discount_factor: float = 1.0, alpha: float = 0.1,
                 epsilon: float = 0.1):
        self.env_nS = env_nS
        self.env_nA = env_nA
        # memory_size and epsilon accepted for config compat but unused
        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self._init_structures()

    def _init_structures(self) -> None:
        self.H = np.zeros((self.env_nS, self.env_nA))
        self._transitions = np.ones((self.env_nS, 2)) * -1
        self._transitions[:, 0] = np.arange(self.env_nS)
        self._trajectory_buffer: list = []

    def act(self, state: int) -> int:
        probs = softmax(self.H[state])
        return np.random.choice(self.env_nA, p=probs)

    def reset(self) -> None:
        self._init_structures()
        self._trajectory_buffer = []

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

    def _run_episode(self, env: VictimEnvironment) -> Dict[str, Any]:
        # 1. Collect full trajectory
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        trajectory = []  # list of (state, action, reward)
        episode_reward = 0.0

        max_steps = _get_max_steps(env)
        for t in range(max_steps):
            action = self.act(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result if len(result) == 5 else (*result[:3], False, result[3])
            done = terminated or truncated
            self._transitions[state, 1] = action
            trajectory.append((state, action, reward))
            episode_reward += reward
            if done:
                break
            state = next_state

        # 2. Compute Monte Carlo returns G_t backwards
        G = 0.0
        returns = []
        for _, _, r in reversed(trajectory):
            G = r + self.discount_factor * G
            returns.insert(0, G)

        # 3. REINFORCE update
        for (s_t, a_t, _), G_t in zip(trajectory, returns):
            probs = softmax(self.H[s_t])
            # grad log pi(a_t | s_t) for softmax = I[a_t] - pi(s_t)
            grad = -probs.copy()
            grad[a_t] += 1.0
            self.H[s_t] += self.alpha * G_t * grad

        # Collect (pre_state, pre_action, state, action) for trajectory encoder
        for i in range(1, len(trajectory)):
            pre_s, pre_a, _ = trajectory[i - 1]
            s, a, _ = trajectory[i]
            self._trajectory_buffer.append((pre_s, pre_a, s, a))

        return {
            'reward': episode_reward,
            'length': len(trajectory),
            'transitions': [(s, a) for s, a, _ in trajectory],
        }

    def get_policy_matrix(self) -> np.ndarray:
        return self.H

    def get_behavior_trace(self) -> np.ndarray:
        return self._transitions.copy()

    def save(self, path: str) -> None:
        np.save(f"{path}_preferences.npy", self.H)

    def load(self, path: str) -> None:
        self.H = np.load(f"{path}_preferences.npy")
