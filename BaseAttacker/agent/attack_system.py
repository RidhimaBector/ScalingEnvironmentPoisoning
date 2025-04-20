from typing import Tuple
import numpy as np
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch

from agent.system import System
from algorithms.algorithm import Algorithm
from envs.attack_environment import AttackEnvironment
from constants import WHITEBOX_METRICS, BLACKBOX_METRICS, PARTIAL_BLACKBOX_METRICS
from utils.attack_cost import AttackCost
from utils.utils_attack import *
from agent.victim_system import PrivacyMode
from ae.encoder_service import EncoderService, EncoderType

METRICS = WHITEBOX_METRICS

DEFAULT_NUM_EPISODES = 50

class AttackSystem(System):
    """
    AttackSystem class that manages attack agent training and evaluation
    """
    def __init__(self, env: AttackEnvironment, algorithm: Algorithm, args, model_dir, config):
        super().__init__(env, algorithm)
        self.model_dir = model_dir
        self.args = args
        self.ddpg_loss = []
        self.buffer = self._setup_replay_buffer()
        self.experiment_time = time.time()
        self._encoder_service = EncoderService(
            config,
            env=self.env.victim_system.env,
            algorithm=self.env.victim_system.algorithm,
            encoder_type=EncoderType.COMBINED
        )
        # Initialize all metrics tracking
        self.buffer_metrics = {metric: [] for metric in METRICS}

        # Initialize model statistics tracking
        self.model_data = []
        self.model_good_data = []
        self.model_bad_data = []
        self.best_model_stats = None
        self.worst_model_stats = None

        # Initialize metrics DataFrame
        self.metrics_df = pd.DataFrame(columns=['episode'] + METRICS).astype({col: 'float64' for col in METRICS})
        self.metrics_df['episode'] = self.metrics_df['episode'].astype('int64')  # Keep episode as integer

        # Initialize target policy
        self.target = self._initialize_target_policy()

    @property
    def encoder_service(self):
        return self._encoder_service

    @encoder_service.setter
    def encoder_service(self, value):
        self._encoder_service = value

    @property
    def env(self) -> AttackEnvironment:
        return self._env

    def reset(self) -> Tuple[AttackEnvironment, Algorithm]:
        """Reset the attack environment and algorithm"""
        self.env.reset()
        self.algorithm.reset()
        return self.env, self.algorithm

    def attack_effort(self, env_dynamics):
        return self.env.Attack_Effort(env_dynamics)

    def attack_cost_compute_K(self, target, cost_matrix):
        attack_cost = AttackCost()

        # With transition model if available
        if hasattr(self.env.victim_system.env, 'T'):
            return attack_cost.kullback_leibler_divergence(
                self.env.victim_system.algorithm.Q,
                target,
                transition_model=self.env.victim_system.env.T
            )

        # With policy approximator (encoder)
        def policy_approximator(policy):
            # Use encoder to approximate policy
            return self.env.encoder_service.Policy_Embedding(policy)

        return attack_cost.kullback_leibler_divergence(
            self.env.victim_system.algorithm.Q,
            target,
            policy_approximator=policy_approximator
        )

    def select_attack_action(self, state, episode):
        """Select action using encoded state representation."""
        if episode < self.args.eps_greedy_start_episodes:
            return self.env.action_space.sample()

        # Convert encoded state tensor to numpy array for algorithm
        state_np = state.detach().numpy().flatten()
        action = self.algorithm.act(state_np)

        return action


    def attack_cost_compute_W(self, target, cost_matrix):
        distance_w = Attack_Cost_Compute_W(self.env.victim_system.env, self.env.victim_system.env.INIT_T, self.env.victim_system.algorithm.Q, target, cost_matrix, distance_type=0)
        distance_grid_w = Attack_Cost_Compute_W(self.env.victim_system.env, self.env.victim_system.env.INIT_T, self.env.victim_system.algorithm.Q, target, cost_matrix, distance_type=1)
        distance_behavior_w = Attack_Cost_Compute_W(self.env.victim_system.env, self.env.victim_system.env.INIT_T, self.env.victim_system.algorithm.Q, target, cost_matrix, distance_type=2)
        return -distance_w, -distance_grid_w, -distance_behavior_w

    def attack_done_identify(self, target):
        return Attack_Done_Identify(target, self.env.victim_system.algorithm.Q)

    def train_system(self):
        for episode in range(self.args.max_episodes):
            self.run_training_episode(
                episode
            )

    def run_training_episode(self, episode: int) -> dict:
        """Run a single training episode."""
        print(f"\n--------- Episode: {episode} ----------")

        self.env.reset()

        self.metrics_df.loc[episode + 1] = 0

        for timestep in range(self.args.max_timesteps):
            timestep_metrics, done = self._run_training_step(
                episode, timestep
            )
            if (done or timestep == self.args.max_timesteps - 1):
                _ = self.update_model_statistics(episode, timestep, timestep_metrics)
                break

        if episode >= self.args.eps_greedy_start_episodes:
            self.ddpg_loss = self.train(episode)

        if (episode + 1) % self.args.eval_freq_episode == 0:
            self.save_checkpoint(self.model_dir, episode + 1)


    def _run_training_step(self, episode: int, timestep: int) -> Tuple[dict, bool]:
        """Execute single training step."""
        start_time = time.time()
        state = self.get_state()
        current_env_dynamics = self.env.victim_system.env.env_dynamics.copy()

        action = self.select_attack_action(state, episode)

        self.env.step(action[:-1])  # Use all components except the last logit value
        x = (action[-1] + 1)/2
        self.env.train_victim(num_episodes=(DEFAULT_NUM_EPISODES*x)+5)

        next_state = self.get_state()

        # METRICS = ["distance_K", "distance_grid_K", "distance_behavior_K",
        #           "distance_W", "distance_grid_W", "distance_behavior_W",
        #           "accuracy", "accuracy_sftmx", "accuracy_sftmx_complete",
        #           "effort", "time"]
        done, timestep_metrics = self.calculate_attack_metrics_with_target(current_env_dynamics, start_time)
        rewards = timestep_metrics['accuracy']

        # Log metrics and add to buffer
        no_episodes = episode + 1
        no_timesteps = timestep + 1
        for metric in METRICS:
            self.buffer_metrics[metric].append([no_episodes, no_timesteps, timestep_metrics[metric]])
            self.metrics_df.loc[no_episodes, metric] += timestep_metrics[metric]

        self.buffer.add(state.view(self.env.nS), action, next_state.view(self.env.nS), rewards, done)

        return timestep_metrics, done

    def get_state(self) -> torch.Tensor:
        """Get encoded state representation from victim system."""
        if self.env.is_initial_state:
            return self._encoder_service.get_initial_state(self.env.victim_system)

        return self._encoder_service.encode_state(
            self.env.victim_system.env.altitude,
            self.env.victim_system.algorithm.transitions
        )

    def train(self, episode):
        """Train the attack policy using the internal replay buffer."""
        return self.algorithm.train(
            self.buffer,
            atk_n_epoch=1,
            atk_n_batch=15,
            batch_size=self.args.batch_size,
            ddpg_loss=self.ddpg_loss,
            i_episode=episode
        )

    def calculate_attack_metrics_with_target(self, current_env_dynamics, start_time):
        """Calculate attack metrics based on privacy mode."""
        privacy_mode = self.env.victim_system.get_privacy_mode()
        metrics = {}

        # Calculate KL divergence distances
        metrics["distance_K"] = -self._compute_kl_distance(distance_type=0)
        metrics["distance_grid_K"] = -self._compute_kl_distance(distance_type=1)
        metrics["distance_behavior_K"] = -self._compute_kl_distance(distance_type=2)

        # Calculate Wasserstein distances
        metrics["distance_W"] = -self._compute_wasserstein_distance(distance_type=0)
        metrics["distance_grid_W"] = -self._compute_wasserstein_distance(distance_type=1)
        metrics["distance_behavior_W"] = -self._compute_wasserstein_distance(distance_type=2)

        # Calculate accuracy and done condition
        target_copy = copy.deepcopy(self.target)
        done, metrics["accuracy"], metrics["accuracy_sftmx"], metrics["accuracy_sftmx_complete"] = \
            Attack_Done_Identify(target_copy, self.env.victim_system.algorithm.Q)

        # Calculate attack effort
        metrics["effort"], current_dynamics = Attack_Effort(
            current_env_dynamics,
            self.env.victim_system.env
        )
        metrics["effort"] = -metrics["effort"]
        metrics["time"] = time.time() - start_time

        return done, metrics

    def calculate_attack_metrics(self, current_env_dynamics, start_time):
        """Calculate attack metrics based on privacy mode."""
        privacy_mode = self.env.victim_system.get_privacy_mode()
        metrics = {}

        # Basic metrics based on privacy mode
        if privacy_mode == PrivacyMode.FULL_WHITEBOX:
            # Calculate KL divergence distances
            metrics["distance_K"] = -self._compute_kl_distance(distance_type=0)
            metrics["distance_grid_K"] = -self._compute_kl_distance(distance_type=1)
            metrics["distance_behavior_K"] = -self._compute_kl_distance(distance_type=2)

            # Calculate Wasserstein distances
            metrics["distance_W"] = -self._compute_wasserstein_distance(distance_type=0)
            metrics["distance_grid_W"] = -self._compute_wasserstein_distance(distance_type=1)
            metrics["distance_behavior_W"] = -self._compute_wasserstein_distance(distance_type=2)

            # Calculate accuracy and done condition
            # target_copy = copy.deepcopy(self.target)
            # done, metrics["accuracy"], metrics["accuracy_sftmx"], metrics["accuracy_sftmx_complete"] = \
            #     Attack_Done_Identify(target_copy, self.env.victim_system.algorithm.Q)

        elif privacy_mode == PrivacyMode.PARTIAL_BLACKBOX:
            metrics["policy_change"] = compute_policy_difference(prev_rep["partial_policy"], new_rep["partial_policy"])
            metrics["trajectory_change"] = compute_trajectory_similarity(prev_rep["trajectories"], new_rep["trajectories"])
        elif privacy_mode == PrivacyMode.FULL_BLACKBOX:
            metrics["embedding_distance"] = np.linalg.norm(prev_rep["embedding"] - new_rep["embedding"])
            metrics["cosine_similarity"] = cosine_similarity(prev_rep["embedding"], new_rep["embedding"])

        # Calculate accuracy and done condition
        target_copy = copy.deepcopy(self.target)
        done, metrics["accuracy"], metrics["accuracy_sftmx"], metrics["accuracy_sftmx_complete"] = \
            Attack_Done_Identify(target_copy, self.env.victim_system.algorithm.Q)

        # Calculate attack effort
        metrics["effort"], current_dynamics = Attack_Effort(
            current_env_dynamics,
            self.env.victim_system.env
        )
        metrics["effort"] = -metrics["effort"]
        metrics["time"] = time.time() - start_time

        return done, metrics

    def _compute_kl_distance(self, distance_type=0):
        """Compute KL divergence between current and target policies."""
        init_T = self.env.victim_system.env.T.copy()
        current_Q = self.env.victim_system.algorithm.Q
        # Get policy from Q-values
        current_policy = get_policy_from_Q(current_Q)

        return Attack_Cost_Compute_K(
            env=self.env.victim_system.env,
            init_T=init_T,
            agent_Q=current_Q,
            target=self.target.copy(),
            policy=current_policy,  # Pass the policy explicitly
            distance_type=distance_type
        )

    def _compute_wasserstein_distance(self, distance_type=0):
        """Compute Wasserstein distance between current and target policies."""
        init_T = self.env.victim_system.env.T.copy()
        current_Q = self.env.victim_system.algorithm.Q

        return Attack_Cost_Compute_W(
            env=self.env.victim_system.env,
            init_T=init_T,
            agent_Q=current_Q,
            target=self.target.copy(),  # Pass a copy to prevent modifications
            distance_type=distance_type
        )

    def update_model_statistics(self, episode, timestep, timestep_metrics):
        """Update best/worst model statistics tracking.

        Args:
            episode (int): Current episode number
            timestep (int): Current timestep within episode
            timestep_metrics (dict): Dictionary of current metrics

        Returns:
            np.ndarray: Current model statistics
        """

        stats = []
        for metric in METRICS:
            stats.extend([
                timestep_metrics[metric],
                self.metrics_df.loc[episode+1, metric]/timestep,
                self.metrics_df.loc[episode+1, metric]
            ])
        cur_model_stats = np.array(stats)

        self.model_data.append([episode+1, timestep+1])
        self.model_data[-1].extend(cur_model_stats.tolist())

        # Initialize stats on first episode
        if episode == 0:
            self.best_model_stats = copy.deepcopy(cur_model_stats)
            self.worst_model_stats = copy.deepcopy(cur_model_stats)
            return cur_model_stats

        # Check if current model is better/worse than previous best/worst
        cur_model_is_good = cur_model_stats >= self.best_model_stats
        cur_model_is_bad = cur_model_stats <= self.worst_model_stats

        if np.sum(cur_model_is_good) > 0:
            # Update good model tracking
            self.model_good_data.append([episode+1, timestep+1])
            self.model_good_data[-1].extend(cur_model_stats.tolist())
            self.model_good_data[-1].extend(self.best_model_stats.tolist())
            self.model_good_data[-1].extend(self.worst_model_stats.tolist())

            # Save good model checkpoint
            self.algorithm.save(f"./{self.model_dir}/good_model_{episode+1}")

            # Update best stats where current model performed better
            self.best_model_stats[cur_model_is_good] = cur_model_stats[cur_model_is_good]

        elif np.sum(cur_model_is_bad) > 0:
            # Update bad model tracking
            self.model_bad_data.append([episode+1, timestep+1])
            self.model_bad_data[-1].extend(cur_model_stats.tolist())
            self.model_bad_data[-1].extend(self.best_model_stats.tolist())
            self.model_bad_data[-1].extend(self.worst_model_stats.tolist())

            # Save bad model checkpoint
            self.algorithm.save(f"./{self.model_dir}/bad_model_{episode+1}")

            # Update worst stats where current model performed worse
            self.worst_model_stats[cur_model_is_bad] = cur_model_stats[cur_model_is_bad]

        return cur_model_stats

    def reset_environment(self, victim_env, victim_algo):
        """Reset the attack environment with updated victim components."""
        self.env.victim_system.env = victim_env
        self.env.victim_system.algorithm = victim_algo
        self.env.reset()

    def _initialize_target_policy(self):
        """Initialize target policy matrix."""
        # Create target policy matrix of shape (nS, nA)
        target = np.zeros((self.env.victim_system.env.nS, self.env.victim_system.env.nA))

        # For each state, create a probability distribution over actions
        for s in range(self.env.victim_system.env.nS):
            # Add small probability to all actions to avoid numerical issues
            target[s] = np.ones(self.env.victim_system.env.nA) * 0.001

            # Set higher probability for the first action (can be modified based on desired target behavior)
            target[s][0] = 1.0

            # Normalize to create valid probability distribution
            target[s] = target[s] / np.sum(target[s])

        return target

    def _get_optimal_action(self, state):
        """Get optimal action for a given state based on environment structure."""
        # This is environment-specific and should be customized
        # Example for a grid world where actions are [UP, RIGHT, DOWN, LEFT]
        env = self.env.victim_system.env
        grid_size = int(np.sqrt(env.nS))
        goal_state = env.nS - 1  # Assuming goal is in bottom-right corner

        current_row = state // grid_size
        current_col = state % grid_size
        goal_row = goal_state // grid_size
        goal_col = goal_state % grid_size

        # Simple heuristic: move towards goal
        if abs(goal_row - current_row) > abs(goal_col - current_col):
            return 0 if goal_row < current_row else 2  # UP or DOWN
        else:
            return 3 if goal_col < current_col else 1  # LEFT or RIGHT

    def save_checkpoint(self, model_dir: str, episode: int):
        """Save training checkpoint and metrics."""
        # Save buffer and policy
        self.buffer.saveBuffer(f"./{model_dir}/{self.experiment_time}")
        self.algorithm.save(f"./{model_dir}/{episode+1}")

        # Create the metrics directory if it doesn't exist
        metrics_dir = f"metrics/{self.experiment_time}"
        os.makedirs(metrics_dir, exist_ok=True)

        # Now save the file
        np.savetxt(f"{metrics_dir}/ddpg_loss.csv", np.array(self.ddpg_loss), delimiter=",")

        # Save other metrics
        for metric in METRICS:
            np.savetxt(f"metrics/{self.experiment_time}/{metric}_buffer.csv", np.array(self.buffer_metrics[metric]), delimiter=",")

        np.savetxt("model_data.csv", np.array(self.model_data), delimiter=",")
        np.savetxt("model_good_data.csv", np.array(self.model_good_data), delimiter=",")
        np.savetxt("model_bad_data.csv", np.array(self.model_bad_data), delimiter=",")

    def _setup_replay_buffer(self):
        """Initialize the replay buffer for the attack agent."""
        state_dim = self.env.nS
        action_dim = self.env.action_space.shape[0]

        from utils.utils_buf import ReplayBuffer
        return ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=int(1e6)  # Standard buffer size of 1 million transitions
        )

    def _compute_trajectory_features(self, trajectories):
        """Convert trajectory samples into feature vectors."""
        # TODO: Implement trajectory feature computation
        pass

    def plot_metrics(self, episode_num):
        # Get metrics for specific episode
        episode_metrics = self.metrics_df[self.metrics_df['episode'] == episode_num]

        # Get average of a metric across all episodes
        metric_mean = self.metrics_df['accuracy'].mean()

        # Plot metric trends
        plt.plot(self.metrics_df['episode'], self.metrics_df['accuracy'])
