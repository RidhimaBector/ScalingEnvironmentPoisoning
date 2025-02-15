"""
SARSA algorithm implementation,
implements Algorithm interface
"""

from typing import Any, Dict

import numpy as np
import utils.utils_buf as utils_buf
from algorithms.algorithm import Algorithm
from envs.victim_environment import VictimEnvironment
from scipy.special import softmax


class SARSA(Algorithm):
    """
    SARSA algorithm implementation,
    implements Algorithm interface
    """


    def __init__(self,
                 env_nS: int,
                 env_nA: int,
                 memory_size: int,
                 discount_factor: float = 1.0,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        """Initialize SARSA agent.

        Args:
            env_nS: Number of states in the environment
            env_nA: Number of actions in the environment
            memory_size: Size of experience replay buffer
            discount_factor: Future reward discount factor
            alpha: Learning rate
            epsilon: Exploration factor
        """
        self.env_nS = env_nS
        self.env_nA = env_nA
        self._init_hyperparameters(memory_size, discount_factor, alpha, epsilon)
        self._init_structures()


    def _init_hyperparameters(self, memory_size: int,
                             discount_factor: float,
                             alpha: float,
                             epsilon: float) -> None:
        """Initialize algorithm hyperparameters."""
        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon


    def _init_structures(self) -> None:
        """Initialize Q-table and memory buffer."""
        self.Q = np.zeros((16, self.env_nA))
        self.memory = self.MEM = utils_buf.Memory(self.memory_size)


    def act(self, state: int) -> int:
        """Select action using current policy."""
        action_probs = softmax(self.Q[state])
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)


    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool) -> None:
        """Update Q-values using SARSA learning."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * self.Q[next_state][next_action]

        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta


    def _run_episode(self, env: VictimEnvironment) -> Dict[str, Any]:
        """Run single training episode.

        Returns:
            Dictionary containing episode statistics
        """
        state = env.reset()
        action = self.act(state)
        episode_reward = 0
        transitions = []

        for t in range(env.max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = self.act(next_state)

            self.update(state, action, reward, next_state, next_action, done)
            transitions.append((state, action))
            episode_reward += reward

            if done:
                break

            state = next_state
            action = next_action

        return {
            'reward': episode_reward,
            'length': t + 1,
            'transitions': transitions
        }


    def train(self, env: VictimEnvironment, num_episodes: int) -> Dict[str, Any]:
        """Train the agent for specified number of episodes.

        Args:
            env: VictimEnvironment
            num_episodes: Number of episodes to train the agent

        Returns:
            Dictionary containing training statistics
        """
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'transitions': self._init_transition_matrix()
        }

        for episode in range(num_episodes):
            episode_stats = self._run_episode(env)
            self._update_stats(stats, episode_stats)

        return stats


    def reset(self) -> None:
        """Reset agent's state."""
        self._init_structures()


    def save(self, path: str) -> None:
        """Save agent's state to disk."""
        np.save(f"{path}_q_table.npy", self.Q)


    def load(self, path: str) -> None:
        """Load agent's state from disk."""
        self.Q = np.load(f"{path}_q_table.npy")


    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize transition tracking matrix."""
        transitions = np.ones((16, 2)) * 4
        transitions[:, 0] = np.arange(16)
        return transitions


    def _update_stats(self, stats: Dict[str, Any],
                     episode_stats: Dict[str, Any]) -> None:
        """Update training statistics."""
        stats['episode_rewards'].append(episode_stats['reward'])
        stats['episode_lengths'].append(episode_stats['length'])
        for state, action in episode_stats['transitions']:
            stats['transitions'][state, 1] = action
