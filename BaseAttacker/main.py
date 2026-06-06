"""Main training script for environment poisoning attack experiments.

Run with Sacred experiment tracking:
    python main.py
    python main.py with max_episodes=50000 batch_size=512
    python main.py with whitebox       # named config
    python main.py with max_episodes=5 max_timesteps=3 victim_n_episodes=3
"""

import copy
import datetime
import os
import time

import numpy as np
import ot
import torch
from gymnasium import spaces

from agent.attack_dispatch import AttackDispatch
from agent.victim_experiment import AttackExperimentTracker, VictimExperimentTracker
from agent.victim_system import VictimSystem, PrivacyMode
from attack.DDPG import DDPG
from attack.PPO import PPO
from ae.ae import AEObservationEncoder
from ae.victim_encoder import VictimEncoder
from ae.lstm_encoder import LSTMTrajectoryEncoder
from ae.observation_encoder import ObservationEncoder
from ae.action_translator import ActionTranslator
from envs.attack_environment import AttackEnvironment
from envs.env3D_4x4 import Grid3D
from victim.victim_Q import VictimQLearning
from victim.victim_sarsa import VictimSARSA
from victim.victim_reinforce import VictimREINFORCE
from experiment import ex, start_victim_sacred, finish_victim_sacred
from yacs.config import CfgNode as CN
import pandas as pd


VICTIM_ALGO_MAP = {
    "qlearning": VictimQLearning,
    "sarsa": VictimSARSA,
    "reinforce": VictimREINFORCE,
}

ATTACK_ALGO_MAP = {
    "ddpg": DDPG,
    "ppo": PPO,
}

ENV_MAP = {
    "grid3d": Grid3D,
}

# YAML config
yaml_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_default.yaml")
with open(yaml_name) as fcfg:
    config = CN.load_cfg(fcfg)
config.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def build_population(args, privacy_mode: PrivacyMode, victim_tracker=None):
    """Create the VictimSystem (unified K-victim manager)."""
    num_victims = getattr(args, 'num_victims', 1)
    algo_class = VICTIM_ALGO_MAP.get(getattr(args, 'victim_algo', 'qlearning'), VictimQLearning)
    env_class = ENV_MAP.get(getattr(args, 'env', 'grid3d'), Grid3D)

    grid_size = getattr(args, 'grid_size', None)
    env_shape = getattr(args, 'env_shape', None)
    if grid_size:
        env_kwargs = {'grid_dimensions': (int(grid_size), int(grid_size))}
    elif env_shape:
        env_kwargs = {'grid_dimensions': tuple(env_shape)}
    else:
        env_kwargs = {}

    algo_kwargs = dict(config.VICTIM.DEFAULT_KWARGS)
    victim_alpha = getattr(args, 'victim_alpha', None)
    if victim_alpha is not None:
        algo_kwargs['alpha'] = victim_alpha

    if num_victims > 1:
        return VictimSystem(
            num_victims=num_victims,
            env_class=env_class,
            env_kwargs=env_kwargs,
            algo_class=algo_class,
            config=config,
            privacy_mode=privacy_mode,
            base_seed=getattr(args, 'seed', 0),
            algo_kwargs=algo_kwargs,
            victim_tracker=victim_tracker,
        )
    else:
        victim_env = env_class(**env_kwargs)
        victim_algo = algo_class(
            victim_env.nS, victim_env.nA, **algo_kwargs
        )
        return VictimSystem(
            env=victim_env,
            algorithm=victim_algo,
            config=config,
            privacy_mode=privacy_mode,
            victim_tracker=victim_tracker,
        )


def build_attack_env(population, args):
    """Create the AttackEnvironment (pure MDP wrapper, no encoder)."""
    return AttackEnvironment(
        victim_population=population,
        attack_dispatch=AttackDispatch(),
        victim_train_episodes=getattr(args, 'victim_n_episodes', 80),
        config=config,
    )


def build_encoders(population, args):
    """Create the standalone ObservationEncoder and ActionTranslator.

    encoder_mode='ae'       → AE(behavior_trace)(5D) + altitude(16D) = 21D (replicates main branch)
    encoder_mode='whitebox' → VictimEncoder: Q(64D) + dynamics(16D) + trace(32D) = 112D
    encoder_mode='lstm'     → LSTMTrajectoryEncoder: LSTM over (s,a) sequences = 128D per victim
    """
    encoder_mode = getattr(args, 'encoder_mode', 'whitebox')

    if encoder_mode == 'ae':
        inner = AEObservationEncoder.from_population(population)
    elif encoder_mode == 'lstm':
        inner = LSTMTrajectoryEncoder.from_population(population, config)
        inner.populate_target_memory(
            population.env,
            population.target,
            n_steps=config.LSTM_ENCODER.SEQ_LEN * 2,
        )
    else:
        inner = VictimEncoder.from_population(population)

    obs_encoder = ObservationEncoder(inner)

    ref_env = population.env
    if hasattr(ref_env, 'perturbation_space'):
        action_space = ref_env.perturbation_space
    else:
        action_space = spaces.Box(-1.0, 1.0, shape=(population.nS,), dtype=np.float64)

    return obs_encoder, ActionTranslator.from_space(action_space)


def build_attack_algo(state_dim, action_dim, max_action, args):
    """Create the attack algorithm (DDPG or PPO)."""
    algo_name = getattr(args, 'attack_algo', 'ddpg')

    if algo_name == 'ppo':
        ppo_kwargs = dict(config.ATTACK.PPO_KWARGS)
        ppo_kwargs.update({
            "nb_states": state_dim,
            "nb_actions": action_dim,
            "max_action": max_action,
        })
        return PPO(**ppo_kwargs)
    else:
        attack_kwargs = dict(config.ATTACK.DEFAULT_KWARGS)
        attack_kwargs.update({
            "nb_states": state_dim,
            "nb_actions": action_dim,
            "max_action": max_action,
            "tau": getattr(args, 'tau', 0.005),
            "discount": getattr(args, 'discount', 0.9),
            "eps_greedy_start_episodes": getattr(args, 'eps_greedy_start_episodes', 30),
            "rate": getattr(args, 'attack_rate', 0.001),
            "prate": getattr(args, 'attack_prate', 0.0001),
        })
        return DDPG(**attack_kwargs)


def _delete_old_checkpoints(model_dir, episode, eval_freq, keep=3):
    """Delete model checkpoint files older than the last `keep` saves."""
    old_episode = episode - keep * eval_freq
    if old_episode <= 0:
        return
    for suffix in ('_actor', '_actor_optimizer', '_critic', '_critic_optimizer'):
        path = os.path.join(model_dir, f"{old_episode}{suffix}")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def train(attack_env, attack_algo, obs_encoder, action_translator,
          args, attack_tracker=None):
    """Main training loop."""
    max_episodes = getattr(args, 'max_episodes', 30000)
    max_timesteps = getattr(args, 'max_timesteps', 15)
    eval_freq = getattr(args, 'eval_freq_episode', 20)
    model_dir = getattr(args, 'model_dir', 'results')
    keep_checkpoints = getattr(args, 'keep_checkpoints', 3)

    num_warmup_episodes = attack_algo.warmup_episodes
    best_accuracy = 0.0

    os.makedirs(model_dir, exist_ok=True)

    for episode in range(max_episodes):
        # Reset all victims
        attack_env.reset()
        obs = obs_encoder.get_initial_embedding()
        episode_reward = 0.0
        episode_start = time.time()
        info = {}
        min_acc_episode = 1.0

        for timestep in range(max_timesteps):
            if episode < num_warmup_episodes:
                raw_action = attack_env.action_space.sample()
            else:
                raw_action = attack_algo.act(np.array(obs))

            action = action_translator.translate(raw_action)
            env_obs, reward, terminated, truncated, info = attack_env.step(action)
            next_obs = obs_encoder.encode(info['victim_results'], env_dynamics=env_obs)

            attack_algo.store_transition(obs, raw_action, next_obs, reward, terminated)
            episode_reward += reward
            min_acc_episode = min(min_acc_episode, info.get('accuracy', 0.0))

            if attack_tracker is not None:
                attack_tracker.log_timestep(episode, timestep, info)

            obs = next_obs
            if terminated:
                break

        if episode >= num_warmup_episodes and attack_algo.ready_to_train():
            attack_algo.update(episode=episode)

        loss_log = attack_algo.get_loss_log()
        episode_metrics = {
            'accuracy': info.get('accuracy', 0.0),
            'min_accuracy': info.get('min_accuracy', 0.0),
            'episode_reward': episode_reward,
            'time': time.time() - episode_start,
        }

        if attack_tracker is not None:
            loss_val = loss_log[-1][2] if loss_log else None
            attack_tracker.log_episode(episode, episode_metrics, loss_val)

        cur_accuracy = info.get('accuracy', 0.0)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            attack_algo.save(os.path.join(model_dir, "best_model"))

        ckpt_marker = ""
        if (episode + 1) % eval_freq == 0:
            attack_algo.save(os.path.join(model_dir, str(episode + 1)))
            _delete_old_checkpoints(model_dir, episode + 1, eval_freq, keep_checkpoints)
            if attack_tracker is not None:
                attack_tracker.log_checkpoint(
                    episode + 1, cur_accuracy, f"{model_dir}/{episode + 1}"
                )
            ckpt_marker = " [ckpt]"

        print(
            f"[{episode+1:{len(str(max_episodes))}d}/{max_episodes}]"
            f"  acc={cur_accuracy:.4f}  min={min_acc_episode:.4f}"
            f"  best={best_accuracy:.4f}  reward={episode_reward:.2f}"
            f"{ckpt_marker}"
        )


# ---------------------------------------------------------------------------
# Sacred entry point
# ---------------------------------------------------------------------------

@ex.main
def sacred_main(_run, _config):
    """Main function with Sacred experiment tracking."""
    print(f"Starting Sacred experiment: {_run._id}")
    print(f"Configuration: {_config}")

    victim_run = start_victim_sacred(_config)
    victim_tracker = VictimExperimentTracker(sacred_run=victim_run)
    attack_tracker = AttackExperimentTracker(sacred_run=_run)

    privacy_mode_str = _config.get('privacy_mode', 'full_whitebox')
    privacy_mode = (
        PrivacyMode.FULL_WHITEBOX
        if privacy_mode_str == 'full_whitebox'
        else PrivacyMode.FULL_BLACKBOX
    )

    model_dir = _config.get('model_dir')
    if model_dir is None:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_dir = f"results/sacred_run_{_run._id}_{current_time}"
    os.makedirs(model_dir, exist_ok=True)
    _run.info['model_dir'] = model_dir

    class Args:
        pass
    args = Args()
    args.model_dir = model_dir
    args.seed = _config.get('seed', 0)
    args.max_episodes = _config.get('max_episodes', 30000)
    args.max_timesteps = _config.get('max_timesteps', 15)
    args.eval_freq_episode = _config.get('eval_freq_episode', 20)
    args.eps_greedy_start_episodes = _config.get('eps_greedy_start_episodes', 30)
    args.batch_size = _config.get('batch_size', 256)
    args.discount = _config.get('discount', 0.9)
    args.tau = _config.get('tau', 0.005)
    args.victim_n_episodes = _config.get('victim_n_episodes', 80)
    args.num_victims = _config.get('num_victims', 1)
    args.victim_algo = _config.get('victim_algo', 'qlearning')
    args.env = _config.get('env', 'grid3d')
    args.env_shape = _config.get('env_shape', None)
    args.grid_size = _config.get('grid_size', None)
    args.attack_algo = _config.get('attack_algo', 'ddpg')
    args.keep_checkpoints = _config.get('keep_checkpoints', 3)
    args.encoder_mode = _config.get('encoder_mode', 'whitebox')
    args.victim_alpha = _config.get('victim_alpha', 0.1)
    args.attack_rate = _config.get('attack_rate', 0.001)
    args.attack_prate = _config.get('attack_prate', 0.0001)

    population = build_population(args, privacy_mode, victim_tracker)
    attack_env = build_attack_env(population, args)
    obs_encoder, action_translator = build_encoders(population, args)

    state_dim = obs_encoder.embedding_dim
    action_dim = attack_env.action_space.shape[0]
    max_action = float(attack_env.action_space.high[0])
    attack_algo = build_attack_algo(state_dim, action_dim, max_action, args)

    attack_env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        train(attack_env, attack_algo, obs_encoder, action_translator,
              args, attack_tracker)

        attack_tracker.log_info('final_episode', args.max_episodes)
        attack_tracker.log_info('attack_algo', args.attack_algo)
        attack_tracker.log_info('num_victims', args.num_victims)

        finish_victim_sacred(victim_run, status='COMPLETED')
    except Exception as e:
        finish_victim_sacred(victim_run, status=f'FAILED: {e}')
        raise


if __name__ == "__main__":
    ex.run_commandline()
