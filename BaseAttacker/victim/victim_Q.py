"""
Q-learning algorithm for victim,
implements Algorithm interface
"""

import copy
import itertools
import os
import sys
from os.path import abspath, dirname
from typing import Any, Dict

import numpy as np
import utils.utils_attack as utils_attack
import utils.utils_buf as utils_buf
from algorithms.algorithm import Algorithm
from envs.victim_environment import VictimEnvironment
from envs.target_def import TARGET
from scipy.special import softmax

if "../" not in sys.path:
    sys.path.append("../")

''' import configuration '''
from yacs.config import CfgNode as CN

yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

#LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
T_max = config.VICTIM.TMAX

class VictimQLearning(Algorithm):
    """
    Q-learning algorithm implementation,
    implements Algorithm interface
    """


    def __init__(self, env_nS: int, env_nA: int, memory_size: int, discount_factor: float = 1.0, alpha: float = 0.1, epsilon: float = 0.1):
        """Initialize Q-Learning agent.

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
        self.Q = np.zeros((16, self.env_nA)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.memory = self.MEM = utils_buf.Memory(MEMORY_SIZE)
        self._transitions = np.ones((16,2)) * 4


    @property
    def transitions(self) -> np.ndarray:
        return self._transitions


    def act(self, state: int) -> int:
        """Select action using current policy."""
        action_probs = softmax(self.Q[state])
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)


    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        """Update Q-values using TD learning."""
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta


    def MakeEpsilonGreedyPolicy(self):
        nA = self.env_nA
        def policy_fn(observation):
            A = np.ones(nA, dtype = float) * self.epsilon/nA
            best_action = np.argmax(self.Q[observation])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn


    def _run_episode_old(self, env: VictimEnvironment, transitions: np.ndarray) -> Dict[str, Any]:
        """Run single training episode.

        Returns:
            Dictionary containing episode statistics
        """
        state = env.reset()
        episode_reward = 0

        max_steps = 1000 if env.max_steps is None else env.max_steps
        for t in range(int(max_steps)):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            transitions[state, 1] = action

            self.update(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            episode_reward += reward

            if done:
                break

        return {
            'reward': episode_reward,
            'length': t + 1,
            'transitions': transitions
        }


    def Train_Model(self, env: VictimEnvironment, num_episodes: int) -> np.ndarray:

        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'transitions': self._init_transition_matrix()
        }

        for episode in range(num_episodes):
            episode_stats = self._run_episode_old(env, stats['transitions'].copy())
            self._update_stats(stats, episode_stats)

        self._transitions = stats['transitions']
        return stats['episode_rewards'], stats['episode_lengths'], stats['transitions']


    def train_for_eval(self, env: VictimEnvironment, num_episodes: int):

        # The policy we're following
        #policy = self.MakeEpsilonGreedyPolicy()
        Q_matrix = self.Q #utils_op.DicQ_To_MatrixQ(self.Q, self.env)
        victim_transitions = np.ones((16,2)) * 4
        victim_transitions[:,0] = np.arange(16)

        #stats_timestep = []
        accuracy_episode = []
        accuracy_softmax_episode = []
        accuracy_softmax_complete_episode = []

        for i_episode in range(num_episodes):
            # logger
            trajectory_list = []
            score = 0

            # Reset env
            state = env.reset()

            for t in itertools.count(): #for t in range(10):
                # logger
                t_sample = []

                # Take a step
                action_probs = softmax(Q_matrix[state]) #action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)
                score += reward
                victim_transitions[state,1] = action

                # TD Update
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

                # save transitions
                """if t!=0:
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)"""

                state = copy.deepcopy(next_state)

                if done:
                    done_attack, accuracy, accuracy_softmax, accuracy_softmax_complete = utils_attack.Attack_Done_Identify(env, TARGET, self.Q)
                    accuracy_episode.append(accuracy)
                    accuracy_softmax_episode.append(accuracy_softmax)
                    accuracy_softmax_complete_episode.append(accuracy_softmax_complete)
                    break

            # Trajectory to MEMORY with head_padding
            """if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = 0
                padding_action = 0
                n_padding = LEN_TRAJECTORY - len(trajectory_list)
                for i in range(n_padding):
                    self.MEM.push(padding_state, padding_action)
                for i in range(len(trajectory_list)):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)
            else:
                for i in range(LEN_TRAJECTORY):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)"""

        return accuracy_episode, accuracy_softmax_episode, accuracy_softmax_complete_episode, victim_transitions #[[0,1,2,3,7,11], :]


    def _run_episode(self, env: VictimEnvironment) -> Dict[str, Any]:
        """Run single training episode."""
        state = env.reset()
        episode_reward = 0
        transitions = []

        for t in range(env.max_steps):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)

            self.update(state, action, reward, next_state, done)
            transitions.append((state, action))
            episode_reward += reward

            if done:
                break
            state = next_state

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
        self.Q = np.zeros((16,self.env_nA)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buf.Memory(self.memory_size)
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
        # Copy over the transitions directly since they're already in the right format
        stats['transitions'] = episode_stats['transitions']

if __name__ == "__main__":
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()

    victim_args = {
        "env": env,
        "MEMORY_SIZE": 100,
        "discount_factor": 1.0,
        "alpha": 0.1,
        "epsilon": 0.1,
    }

    victim = QLearning(**victim_args)
    victim.Train_Model(10)
    victim.Show_PolicyQ()
    victim.Eval_Model()

    print(f"Size of Memory = {victim.MEM.__len__()}")
    print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")

