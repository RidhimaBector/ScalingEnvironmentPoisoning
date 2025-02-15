import argparse
import copy
import datetime
import os
import time

import numpy as np
import ot
import torch
from ae.ae import AutoEncoder
from ae.autoencoder import EnvAutoEncoder
from agent.attack_system import AttackSystem
from agent.victim_system import VictimSystem
from attack.DDPG import DDPG
from constants import METRICS, TARGET, MEM_Target
from envs.environment import Environment
from envs.victim_environment import VictimEnvironment
from envs.attack_environment import AttackEnvironment
from envs.env3D_4x4 import Grid3D
from utils import utils_buf, utils_log
from victim.victim_Q import VictimQLearning
from yacs.config import CfgNode as CN


# YAML config
yaml_name='/Users/kunwarnir/projects/envPoisoning/ScalingEnvironmentPoisoning/BaseAttacker/config/config_default.yaml'
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

SEQ_LEN = config.AE.SEQ_LEN
EMBEDDING_SIZE = config.AE.EMBEDDING_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_cost_matrix(victim_env):
    """Create a cost matrix for the attack environment."""
    grid = np.mgrid[0:4, 0:4].reshape(2,-1).T
    cost_matrix = ot.dist(grid, grid, metric='cityblock') * 20
    np.fill_diagonal(cost_matrix, 10)
    cost_matrix = np.tile(cost_matrix, (victim_env.nA, victim_env.nA))
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="R_0818")               # TensorBoard folder
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--atk_training_start_episodes", default=100, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eps_greedy_start_episodes", default=30, type=int)  # Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=15, type=int)   # Max time steps to run environment
    parser.add_argument("--max_episodes", default=30000, type=int)   # Max episodes to run environment
    parser.add_argument("--eval_freq_episode", default=20, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.9, type=float)     # Discount factor of attack network $\gamma$DDPG with Fixed Discount
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    parser.add_argument("--victim_n_episodes", default=80, type=int)  # number of episodes for victim's updated in poisoned Env
    parser.add_argument("--ae_n_epochs", default=10, type=int)         # number of training epoch

    return parser.parse_args()


def _set_seeds(seed, victim_env, attack_env):
    victim_env.seed(seed)
    attack_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _save_attack_policy(model_dir, no_episodes, Buffer, Policy, ddpg_loss, buffer_metrics, model_data, model_good_data, model_bad_data):
    Buffer.saveBuffer(f"./{model_dir}/")
    Policy.save(f"./{model_dir}/{no_episodes}")
    np.savetxt("ddpg_loss.csv", np.array(ddpg_loss), delimiter=",")
    for metric in METRICS:
        np.savetxt(f"{metric}_buffer.csv", np.array(buffer_metrics[metric]), delimiter=",")
    np.savetxt("model_data.csv", np.array(model_data), delimiter=",")
    np.savetxt("model_good_data.csv", np.array(model_good_data), delimiter=",")
    np.savetxt("model_bad_data.csv", np.array(model_bad_data), delimiter=",")


def setup_model_dir(args):
    """Create and setup the model directory for saving results."""
    if args.model_dir is None:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.model_dir = f"results/{current_time}"

    # Create directories if they don't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    return args.model_dir


def setup_replay_buffer(attack_env):
    """Initialize the replay buffer for the attack agent."""
    state_dim = attack_env.observation_space.shape[0]
    action_dim = attack_env.action_space.shape[0]

    # Initialize replay buffer with state and action dimensions
    return ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=int(1e6)  # Standard buffer size of 1 million transitions
    )


def setup_autoencoder(victim_env):
    """Initialize and setup the environment autoencoder."""
    ae_kwargs = {
        "env": victim_env,
        "enc_in_size": config.AE.ENC_IN_SIZE,
        "enc_out_size": config.AE.ENC_OUT_SIZE,
        "enc_num_layer": config.AE.ENC_NUM_LAYER,
        "dec_fc_in_size": config.AE.DEC_FC_IN_SIZE,
        "dec_fc_out_size": config.AE.DEC_FC_OUT_SIZE,
        "dec_lstm_in_size": config.AE.DEC_LSTM_IN_SIZE,
        "dec_lstm_out_size": config.AE.DEC_LSTM_OUT_SIZE,
        "dec_lstm_num_layer": config.AE.DEC_LSTM_NUM_LAYER,
        "seq_len": SEQ_LEN,
        "embedding_len": EMBEDDING_SIZE,
        "n_epochs": config.AE.N_EPOCHS,
        "lr": config.AE.LEARNING_RATE
    }
    return EnvAutoEncoder(**ae_kwargs)


def save_checkpoint(model_dir, no_episodes, buffer, policy, ddpg_loss, buffer_metrics, model_data, model_good_data, model_bad_data):
    """Save training checkpoint and metrics."""
    buffer.saveBuffer(f"./{model_dir}/")
    policy.save(f"./{model_dir}/{no_episodes}")

    # Save metrics
    np.savetxt("ddpg_loss.csv", np.array(ddpg_loss), delimiter=",")
    for metric in METRICS:
        np.savetxt(f"{metric}_buffer.csv", np.array(buffer_metrics[metric]), delimiter=",")
    np.savetxt("model_data.csv", np.array(model_data), delimiter=",")
    np.savetxt("model_good_data.csv", np.array(model_good_data), delimiter=",")
    np.savetxt("model_bad_data.csv", np.array(model_bad_data), delimiter=",")

"""
Main training loop for the environment poisoning attack. The process follows:

1. Setup Phase:
    - Initialize victim components (3D grid environment & Q-learning)
    - Initialize attack components (attack environment & DDPG policy)
    - Load pretrained autoencoder and setup replay buffer

2. Training Loop (for each episode):
    - Reset environments
    - For each timestep:
        a. Select attack action (random during warmup, then from policy)
        b. Execute attack and train victim
        c. Calculate attack costs and metrics
        d. Store experience in replay buffer

    - After episode:
        - Train attack policy (if past warmup)
        - Periodically save models and metrics

3. Metrics Tracked:
    - Attack effectiveness (KL divergence, Wasserstein distance)
    - Attack accuracy
    - Environmental modification effort
    - Training time

Returns:
    None (saves models and metrics to specified directory)
"""
def main():
    # Parse arguments and setup
    args = parse_arguments()

    # Initialize environments and systems
    victim_system, attack_system = setup_systems(args)

    # Setup training components
    model_dir = setup_model_dir(args)
    buffer = setup_replay_buffer(attack_system.env)
    ae = setup_autoencoder(victim_system.env)

    #attack_system.train_system()

    # Training loop
    for episode in range(args.max_episodes):
        temp_metrics, cumulative_metrics = attack_system.run_training_episode(
            episode, victim_system, buffer, args.max_timesteps
        )

        # Train attack policy if past warmup
        if episode >= args.eps_greedy_start_episodes:
            attack_system.train(buffer, args)

        # Save periodically
        if (episode + 1) % args.eval_freq_episode == 0:
            save_checkpoint(
                model_dir, episode + 1, buffer, attack_system,
                attack_system.buffer_metrics,
                attack_system.model_data,
                attack_system.model_good_data,
                attack_system.model_bad_data
            )

def setup_systems(args):
    # Initialize victim components
    victim_env = Grid3D()
    victim_algo = VictimQLearning(victim_env.nS, victim_env.nA, **config.VICTIM.DEFAULT_KWARGS)
    victim_system = VictimSystem(victim_env, victim_algo)

    # Initialize attack components
    attack_env = AttackEnvironment(victim_system)
    attack_system = setup_attack_system(attack_env, args)

    _set_seeds(args.seed, victim_env, attack_env)

    return victim_system, attack_system

def setup_attack_system(attack_env, args):
    attack_kwargs = config.ATTACK.DEFAULT_KWARGS.copy()
    attack_kwargs.update({
        "nb_states": attack_env.nS,
        "nb_actions": attack_env.action_space.shape[0],
        "max_action": float(attack_env.action_space.high[0]),
        "tau": args.tau,
        "discount": args.discount,
    })

    attack_algo = DDPG(**attack_kwargs)
    env_ae = setup_autoencoder(victim_env)

    return AttackSystem(attack_env, attack_algo, env_ae)

if __name__ == "__main__":
    main()
