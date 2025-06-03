from typing import Tuple
import numpy as np
import copy
import time

from agent.system import System
from algorithms.algorithm import Algorithm
from envs.attack_environment import AttackEnvironment
from ae.autoencoder import EnvAutoEncoder
from constants import METRICS
from utils.attack_cost import AttackCost
from utils.utils_attack import *
from envs.target_def import MEM_Target
from utils.utils_buf import Memory
from ae.encoder_service import EncoderService

class AttackSystem(System):
    """
    AttackSystem class that manages attack agent training and evaluation
    """
    def __init__(self, env: AttackEnvironment, algorithm: Algorithm, args, model_dir, config):
        super().__init__(env, algorithm)
        self._encoder_service = self._setup_encoder_service(env.victim_env, config)
        self.initial_dynamics = env.victim_env.env_dynamics.copy()
        self.eps_greedy_start_episodes = args.eps_greedy_start_episodes
        self.model_dir = model_dir
        self.args = args

        # Initialize replay buffer
        self.buffer = self._setup_replay_buffer()

        # Initialize metrics tracking
        self.buffer_metrics = {metric: [] for metric in METRICS}
        self.model_data = []
        self.model_good_data = []
        self.model_bad_data = []

        # Initialize target policy with explicit dimensions
        nS = env.victim_env.nS
        nA = env.victim_env.nA
        self.target = np.zeros((nS, nA))

        # Initialize with uniform distribution and small epsilon
        epsilon = 0.001
        for s in range(nS):
            self.target[s] = np.ones(nA) * epsilon
            self.target[s][0] = 1.0  # Prefer first action
            self.target[s] /= np.sum(self.target[s])  # Normalize

        # Verify target shape
        assert self.target.shape == (nS, nA), f"Target shape mismatch: expected ({nS}, {nA}), got {self.target.shape}"

        # Initialize target policy
        self._initialize_target_policy()

    @property
    def env(self) -> AttackEnvironment:
        return self._env

    @property
    def encoder_service(self):
        return self._encoder_service

    @encoder_service.setter
    def encoder_service(self, value):
        self._encoder_service = value

    def reset(self) -> Tuple[AttackEnvironment, Algorithm]:
        """Reset the attack environment and algorithm"""
        self.env.reset()
        self.algorithm.reset()
        return self.env, self.algorithm


    def attack_effort(self, env_dynamics):
        return self.env.Attack_Effort(env_dynamics)


    def train_victim(self, victim_num_episodes : int = 50, ae_num_episodes : int = 1):
        self.encoder_service.n_epochs = ae_num_episodes
        for episode in range(victim_num_episodes):
            self.env.victim_agent.Train_Model(1)
            if self.env.victim_agent.MEM.__len__() >= self.encoder_service.memory_size:
                self.encoder_service._train(self.env.victim_agent.MEM, MEM_Target)


    def attack_cost_compute_K(self, target, cost_matrix):
        attack_cost = AttackCost()

        # With transition model if available
        if hasattr(self.env.victim_env, 'T'):
            return attack_cost.kullback_leibler_divergence(
                self.env.victim_algo.Q,
                target,
                transition_model=self.env.victim_env.T
            )

        # With policy approximator (encoder)
        def policy_approximator(policy):
            # Use encoder to approximate policy
            return self.encoder_service.Policy_Embedding(policy)

        return attack_cost.kullback_leibler_divergence(
            self.env.victim_algo.Q,
            target,
            policy_approximator=policy_approximator
        )

    def select_attack_action(self, state, episode):
        return self.algorithm.act(np.array(state), episode, self.eps_greedy_start_episodes)


    def attack_cost_compute_W(self, target, cost_matrix):
        distance_w = Attack_Cost_Compute_W(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=0)
        distance_grid_w = Attack_Cost_Compute_W(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=1)
        distance_behavior_w = Attack_Cost_Compute_W(self.env.victim_env, self.env.victim_env.INIT_T, self.env.victim_algo.Q, target, cost_matrix, distance_type=2)
        return -distance_w, -distance_grid_w, -distance_behavior_w


    def attack_done_identify(self, target):
        return Attack_Done_Identify(target, self.env.victim_algo.Q)


    def step(self, action, current_env_dynamics):
        # Execute action and get next state
        self.env.step(action)

        # Train victim in modified environment
        rewards, lengths, transitions = self.env.victim_system.algorithm.Train_Model(
            env=self.env.victim_env,  # Pass the environment
            num_episodes=80
        )

        # # Get combined state representation
        # state = self.encoder_service.encode_state(
        #     self.env.victim_env.altitude,
        #     transitions
        # )

        # Calculate metrics - pass victim_system instead of transitions
        metrics = self.calculate_attack_metrics(self.env.victim_system, current_env_dynamics)

        return "state", metrics['reward'], metrics['done'], metrics

    def train(self, args):
        """Train the attack policy using the internal replay buffer."""
        return self.algorithm.train(
            self.buffer,
            atk_n_epoch=1,
            atk_n_batch=15,
            batch_size=args.batch_size
        )

    def calculate_attack_metrics(self, victim_system, current_env_dynamics):
        """Calculate all attack-related metrics."""
        # Verify target is properly initialized before use
        assert self.target is not None and self.target.ndim == 2, f"Invalid target shape: {self.target.shape if self.target is not None else None}"

        metrics = {}

        # Calculate KL divergence distances
        metrics["distance_K"] = -self._compute_kl_distance(victim_system, distance_type=0)
        metrics["distance_grid_K"] = -self._compute_kl_distance(victim_system, distance_type=1)
        metrics["distance_behavior_K"] = -self._compute_kl_distance(victim_system, distance_type=2)

        # Calculate Wasserstein distances
        metrics["distance_W"] = -self._compute_wasserstein_distance(victim_system, distance_type=0)
        metrics["distance_grid_W"] = -self._compute_wasserstein_distance(victim_system, distance_type=1)
        metrics["distance_behavior_W"] = -self._compute_wasserstein_distance(victim_system, distance_type=2)

        # Calculate accuracy and done condition
        done, metrics["accuracy"], metrics["accuracy_sftmx"], metrics["accuracy_sftmx_complete"] = \
            Attack_Done_Identify(self.env.victim_env, self.target.copy(), victim_system.algorithm.Q)

        # Calculate attack effort
        metrics["effort"], current_dynamics = Attack_Effort(
            current_env_dynamics,
            self.env.victim_env
        )
        metrics["effort"] = -metrics["effort"]

        return done, metrics

    def _compute_kl_distance(self, victim_system, distance_type=0):
        """Compute KL divergence between current and target policies."""
        init_T = self.env.victim_env.T.copy()
        current_Q = victim_system.algorithm.Q
        # Get policy from Q-values
        current_policy = get_policy_from_Q(current_Q)

        return Attack_Cost_Compute_K(
            env=self.env.victim_env,
            init_T=init_T,
            agent_Q=current_Q,
            target=self.target.copy(),
            policy=current_policy,  # Pass the policy explicitly
            distance_type=distance_type
        )

    def get_target_policy(self):
        """Get the target policy for the attack.
        This should return the policy matrix we want the victim to learn.
        """
        # This is a placeholder - implement based on your specific target policy
        target = np.zeros((self.env.victim_env.nS, self.env.victim_env.nA))
        # Fill target with your desired policy values
        return target

    def _compute_wasserstein_distance(self, victim_system, distance_type=0):
        """Compute Wasserstein distance between current and target policies."""
        init_T = self.env.victim_env.T.copy()
        current_Q = victim_system.algorithm.Q

        return Attack_Cost_Compute_W(
            env=self.env.victim_env,
            init_T=init_T,
            agent_Q=current_Q,
            target=self.target.copy(),  # Pass a copy to prevent modifications
            distance_type=distance_type
        )


    def train_system(self):

        for episode in range(self.args.max_episodes):
            _, _ = self.run_training_episode(
                episode, self.args.max_timesteps
            )
#================================================================================================
            if episode >= self.args.eps_greedy_start_episodes:
                self.train()

            if (episode + 1) % self.args.eval_freq_episode == 0:
                self.save_checkpoint(self.model_dir, episode + 1)


    def run_training_episode(self, episode: int, num_timesteps: int):
        """Run a single training episode."""
        print(f"\n--------- Episode: {episode} ----------")

        cumulative_metrics = temp_metrics = {metric: 0 for metric in METRICS}

        self.env.reset()

        # Reset systems
        # victim_env, victim_algo = self.env.victim_system.reset()
        # self.reset_environment(victim_env, victim_algo)


        x = self.env.get_state(self.encoder_service)


        current_env_dynamics = self.env.victim_env.env_dynamics.copy()

        for timestep in range(num_timesteps):
  #================================================================================================
            temp_metrics = self._run_training_step(
                episode, timestep, self.env.victim_system,
                x, current_env_dynamics, cumulative_metrics
            )

            if temp_metrics['done']:
                break

        return temp_metrics, cumulative_metrics


    def _run_training_step(self, episode, timestep, victim_system,
                         state, current_env_dynamics, cumulative_metrics):
        """Execute single training step."""
        start_time = time.time()

        # Get action and execute
        action = self.select_attack_action(state, episode)
#================================================================================================

        next_state, reward, done, temp_metrics = self.step(
            action, victim_system, current_env_dynamics
        )

        # Log metrics
        no_episodes = episode + 1
        no_timesteps = timestep + 1

        for metric in METRICS:
            self.buffer_metrics[metric].append([no_episodes, no_timesteps, temp_metrics[metric]])
            cumulative_metrics[metric] += temp_metrics[metric]

        # Store transition
        self.buffer.add(previous_state, action, next_state, reward, done)

        return temp_metrics

    def update_model_statistics(self, episode, timestep, temp_metrics, cumulative_metrics):
        """Update best/worst model statistics."""
        stats = []
        for metric in METRICS:
            stats.extend([
                temp_metrics[metric],
                cumulative_metrics[metric]/timestep,
                cumulative_metrics[metric]
            ])
        cur_model_stats = np.array(stats)

        self._update_best_worst_models(episode, timestep, cur_model_stats)
        return cur_model_stats

    def _update_best_worst_models(self, episode, timestep, cur_model_stats):
        """Update tracking of best and worst performing models."""
        if episode == 0:
            self.best_model_stats = copy.deepcopy(cur_model_stats)
            self.worst_model_stats = copy.deepcopy(cur_model_stats)
            return

        cur_model_is_good = cur_model_stats >= self.best_model_stats
        cur_model_is_bad = cur_model_stats <= self.worst_model_stats

        if np.sum(cur_model_is_good) > 0:
            self._update_good_model(episode, timestep, cur_model_stats, cur_model_is_good)
        elif np.sum(cur_model_is_bad) > 0:
            self._update_bad_model(episode, timestep, cur_model_stats, cur_model_is_bad)

    def _setup_encoder_service(self, victim_env, config):
        """Initialize and setup the encoder service."""
        return EncoderService(victim_env, config)

    def reset_environment(self, victim_env, victim_algo):
        """Reset the attack environment with updated victim components."""
        self.env.victim_env = victim_env
        self.env.victim_algorithm = victim_algo
        self.env.reset()

    def _initialize_target_policy(self):
        """Initialize target policy matrix."""
        # Create target policy matrix of shape (nS, nA)
        target = np.zeros((self.env.victim_env.nS, self.env.victim_env.nA))

        # For each state, create a probability distribution over actions
        for s in range(self.env.victim_env.nS):
            # Add small probability to all actions to avoid numerical issues
            target[s] = np.ones(self.env.victim_env.nA) * 0.001

            # Set higher probability for the first action (can be modified based on desired target behavior)
            target[s][0] = 1.0

            # Normalize to create valid probability distribution
            target[s] = target[s] / np.sum(target[s])

        self.target = target

    def _get_optimal_action(self, state):
        """Get optimal action for a given state based on environment structure."""
        # This is environment-specific and should be customized
        # Example for a grid world where actions are [UP, RIGHT, DOWN, LEFT]
        env = self.env.victim_env
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
        self.env.victim_algo.saveBuffer(f"./{model_dir}/")
        self.algorithm.save(f"./{model_dir}/{episode}")

        # Save metrics
        np.savetxt("ddpg_loss.csv", np.array(self.ddpg_loss), delimiter=",")
        for metric in METRICS:
            np.savetxt(f"{metric}_buffer.csv", np.array(self.buffer_metrics[metric]), delimiter=",")
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
