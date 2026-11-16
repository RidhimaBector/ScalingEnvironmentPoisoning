"""Main training script for environment poisoning attack experiments.

Run with Sacred experiment tracking:
    python main.py
    python main.py with max_episodes=50000 batch_size=512
    python main.py with whitebox       # named config
    python main.py with max_episodes=5 max_timesteps=3 victim_n_episodes=3
"""

import copy
import datetime
import json
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


from registry import ENV_MAP, VICTIM_ALGO_MAP  # noqa: E402  (name -> class)

ATTACK_ALGO_MAP = {
    "ddpg": DDPG,
    "ppo": PPO,
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
        env_kwargs = dict(getattr(args, 'env_kwargs', None) or {})

    if hasattr(algo_class, 'from_env'):
        # Function-approximation victims size themselves from the environment
        algo_kwargs = dict(getattr(args, 'victim_kwargs', None) or {})
    else:
        algo_kwargs = dict(config.VICTIM.DEFAULT_KWARGS)
        victim_alpha = getattr(args, 'victim_alpha', None)
        if victim_alpha is not None:
            algo_kwargs['alpha'] = victim_alpha
        algo_kwargs.update(getattr(args, 'victim_kwargs', None) or {})

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
        if hasattr(algo_class, 'from_env'):
            victim_algo = algo_class.from_env(victim_env, **algo_kwargs)
        else:
            victim_algo = algo_class(victim_env.nS, victim_env.nA, **algo_kwargs)
        return VictimSystem(
            env=victim_env,
            algorithm=victim_algo,
            config=config,
            privacy_mode=privacy_mode,
            victim_tracker=victim_tracker,
        )


def build_attack_env(population, args, encoder=None):
    """Create the AttackEnvironment, optionally with injected encoder."""
    return AttackEnvironment(
        victim_population=population,
        attack_dispatch=AttackDispatch(),
        victim_train_episodes=getattr(args, 'victim_n_episodes', 80),
        config=config,
        encoder=encoder,
        reward_mode=getattr(args, 'reward_mode', 'trace'),
        obs_dynamics=getattr(args, 'obs_dynamics', 'actions'),
        effort_weight=getattr(args, 'effort_weight', 0.0),
    )


def build_encoders(population, args):
    """Create the standalone ObservationEncoder and ActionTranslator.

    encoder_mode='ae'       → AE(behavior_trace)(5D) + altitude(16D) = 21D (replicates main branch)
    encoder_mode='whitebox' → VictimEncoder: Q(64D) + dynamics(16D) + trace(32D) = 112D
    encoder_mode='lstm'     → LSTMTrajectoryEncoder: LSTM over (s,a) sequences = 128D per victim
    """
    encoder_mode = getattr(args, 'encoder_mode', 'whitebox')
    env = population.env
    # Context = the non-behaviour part of the observation (see AttackEnvironment.obs_dynamics)
    obs_dynamics = getattr(args, 'obs_dynamics', 'actions')
    norm = getattr(args, 'normalize_obs', False)
    if obs_dynamics == 'env':
        context_dim = env.dynamics_dim
        bounds = env.dynamics_bounds() if (norm and hasattr(env, 'dynamics_bounds')) else None
        initial_context = env.get_dynamics() if norm else None
    elif obs_dynamics == 'actions':
        context_dim = int(np.prod(env.perturbation_space.shape))
        bounds, initial_context = None, None     # running sum starts at zero
    else:
        context_dim, bounds, initial_context = 0, None, None

    if encoder_mode == 'ae':
        inner = AEObservationEncoder.from_population(
            population, model_path=getattr(args, 'ae_model_path', None),
            context_dim=context_dim, context_bounds=bounds, initial_context=initial_context)
    elif encoder_mode == 'lstm':
        inner = LSTMTrajectoryEncoder.from_population(
            population, config,
            embedding_dim=getattr(args, 'lstm_embedding_dim', None),
            seq_len=getattr(args, 'lstm_seq_len', None),
            memory_size=getattr(args, 'lstm_memory_size', None))
        if hasattr(env, 'target_trajectories'):
            # Target given as recorded behaviour (e.g. the lander)
            inner.set_target_trajectories(env.target_trajectories())
        else:
            inner.populate_target_memory(
                env, population.target, n_steps=inner._seq_len * 2)
    else:
        inner = VictimEncoder.from_population(population)

    # LSTM embedding carries no information about the current altitude; optionally
    # append it (the attacker knows the perturbations it has applied).
    append_dyn = encoder_mode == 'lstm' and getattr(args, 'lstm_append_dynamics', True)
    obs_encoder = ObservationEncoder(
        inner, append_dynamics_dim=context_dim if append_dyn else 0,
        normalize_dynamics=bounds is not None, dynamics_bounds=bounds,
        initial_dynamics=initial_context if append_dyn else None)

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
            "updates_per_step": getattr(args, 'updates_per_step', getattr(config.ATTACK.DEFAULT_KWARGS, 'updates_per_step', 1)),
            "bsize": getattr(args, 'batch_size', 256),
            "expl_noise": getattr(args, 'expl_noise', 0.0),
            "td3": getattr(args, 'td3', False),
            "bc_weight": getattr(args, 'bc_weight', 0.0),
        })
        return DDPG(**attack_kwargs)


def _delete_old_checkpoints(model_dir, episode, eval_freq, keep=3):
    """Delete model checkpoint files older than the last `keep` saves."""
    old_episode = episode - keep * eval_freq
    if old_episode <= 0:
        return
    for suffix in ('_actor', '_actor_optimizer', '_critic', '_critic_optimizer',
                   '_critic2', '_critic2_optimizer'):
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
    best_accuracy = 0.0   # best 50-episode mean episode reward so far
    recent_rewards = []

    os.makedirs(model_dir, exist_ok=True)
    attack_env.victim_system.env.reset_dynamics()
    initial_dynamics = np.asarray(attack_env.victim_system.env.get_dynamics(), dtype=float).copy()

    # Adaptive push (cross-entropy method): per-cell probabilities of pushing
    # down / hold / up, moved towards the push episodes with the highest reward.
    push_cem = getattr(args, 'push_cem', False)
    cem_probs = np.full(attack_env.action_space.shape + (3,), 1.0 / 3.0)
    cem_batch = []                      # (episode_reward, direction index per cell)
    CEM_BATCH = getattr(args, 'cem_batch', 16)
    CEM_ELITE = getattr(args, 'cem_elite', 4)
    CEM_LR = getattr(args, 'cem_lr', 0.5)
    # Probability floor per direction. A fixed 0.05 means each cell leaves its best
    # direction 10% of the time, so on 36 cells only ~2% of pushes are the best map.
    # cem_floor < 0 scales it with the grid: 0.25 / n_cells per direction.
    CEM_FLOOR = getattr(args, 'cem_floor', 0.05)
    if CEM_FLOOR < 0:
        CEM_FLOOR = 0.25 / int(np.prod(attack_env.action_space.shape))
    # Elite episodes for the imitation term: top elite_k by episode reward.
    bc_weight = getattr(args, 'bc_weight', 0.0)
    elite_k = getattr(args, 'elite_k', 20)
    elites = []                         # (episode_reward, obs list, action list)

    for episode in range(max_episodes):
        # Reset all victims; obs comes from the env (encoder injected)
        obs, _ = attack_env.reset()
        episode_reward = 0.0
        episode_start = time.time()
        info = {}
        min_acc_episode = 1.0
        greedy_sum = path_sum = 0.0
        n_steps = 0

        # Structured exploration: in a "push" episode every cell is pushed in one fixed
        # random direction (up / down / hold) for the whole episode, so the replay buffer
        # covers altitude maps far from the natural one. Plain per-step random actions
        # average out to ~no change. Used for half of the warm-up episodes and for a
        # fraction explore_push_prob of later episodes.
        push_prob = getattr(args, 'explore_push_prob', 0.0)
        warm = episode < num_warmup_episodes
        push = push_prob > 0 and np.random.rand() < (0.5 if warm else push_prob)
        if push_cem:
            push_idx = np.array([np.random.choice(3, p=pr) for pr in cem_probs])
        else:
            push_idx = np.random.choice(3, size=attack_env.action_space.shape)
        push_dir = push_idx.astype(float) - 1.0
        ep_obs, ep_act = [], []

        for timestep in range(max_timesteps):
            if push:
                raw_action = np.clip(
                    push_dir + np.random.normal(0.0, 0.2, size=push_dir.shape), -1.0, 1.0)
            elif warm:
                raw_action = attack_env.action_space.sample()
            else:
                raw_action = attack_algo.act(np.array(obs))

            action = action_translator.translate(raw_action)
            next_obs, reward, terminated, truncated, info = attack_env.step(action)

            attack_algo.store_transition(obs, raw_action, next_obs, reward, terminated)
            ep_obs.append(np.array(obs, dtype=np.float32).reshape(-1))
            ep_act.append(np.array(raw_action, dtype=np.float32).reshape(-1))
            episode_reward += reward
            min_acc_episode = min(min_acc_episode, info.get('accuracy', 0.0))
            greedy_sum += info.get('greedy_accuracy', 0.0)
            path_sum += info.get('path_match', 0.0)
            n_steps += 1

            if attack_tracker is not None:
                attack_tracker.log_timestep(episode, timestep, info)

            obs = next_obs
            if terminated:
                break

        if push and push_cem:
            cem_batch.append((episode_reward, push_idx))
            if len(cem_batch) >= CEM_BATCH:
                top = sorted(cem_batch, key=lambda x: -x[0])[:CEM_ELITE]
                freq = np.mean([np.eye(3)[idx] for _, idx in top], axis=0)
                cem_probs = (1 - CEM_LR) * cem_probs + CEM_LR * freq
                cem_probs = np.maximum(cem_probs, CEM_FLOOR)
                cem_probs /= cem_probs.sum(axis=1, keepdims=True)
                cem_batch = []

        if bc_weight > 0 and (len(elites) < elite_k or episode_reward > elites[-1][0]):
            elites.append((episode_reward, ep_obs, ep_act))
            elites.sort(key=lambda x: -x[0])
            elites = elites[:elite_k]
            attack_algo.set_elite(
                np.concatenate([np.stack(o) for _, o, _ in elites]),
                np.concatenate([np.stack(a) for _, _, a in elites]))

        # Train encoder once per episode (LSTM only; no-op for other encoders)
        obs_encoder.train_encoder()

        last_losses: dict = {}
        if episode >= num_warmup_episodes and attack_algo.ready_to_train():
            last_losses = attack_algo.update(episode=episode) or {}

        cur_accuracy = info.get('accuracy', 0.0)
        episode_metrics = {
            'accuracy': cur_accuracy,
            'min_accuracy': info.get('min_accuracy', cur_accuracy),
            'episode_reward': episode_reward,
            'time': time.time() - episode_start,
            # Oracle (ground-truth) poisoning success, for evaluation only
            'greedy_accuracy': info.get('greedy_accuracy', 0.0),
            'path_match': info.get('path_match', 0.0),
            'greedy_accuracy_mean': greedy_sum / max(n_steps, 1),
            'path_match_mean': path_sum / max(n_steps, 1),
            # Mean absolute change of the environment's dynamics vector at episode end
            'dynamics_change': float(np.abs(
                attack_env.victim_system.env.get_dynamics() - initial_dynamics).mean()),
        }
        episode_metrics.update(last_losses)  # e.g. critic_loss, actor_loss (algo-specific)

        if attack_tracker is not None:
            # Pass critic_loss as legacy ddpg_loss scalar for backward compat
            loss_val = last_losses.get('critic_loss')
            attack_tracker.log_episode(episode, episode_metrics, loss_val)

        # Best model = best 50-episode moving average of the episode reward
        # (single-episode accuracy is too noisy to select on).
        if not push:
            recent_rewards.append(episode_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_reward = float(np.mean(recent_rewards))
        if episode >= num_warmup_episodes + 50 and avg_reward > best_accuracy:
            best_accuracy = avg_reward
            attack_algo.save(os.path.join(model_dir, "best_model"))
            obs_encoder.save(os.path.join(model_dir, "best_model_encoder.pt"))

        ckpt_marker = ""
        if (episode + 1) % eval_freq == 0:
            attack_algo.save(os.path.join(model_dir, str(episode + 1)))
            obs_encoder.save(os.path.join(model_dir, "latest_encoder.pt"))
            _delete_old_checkpoints(model_dir, episode + 1, eval_freq, keep_checkpoints)
            if attack_tracker is not None:
                attack_tracker.log_checkpoint(
                    episode + 1, cur_accuracy, f"{model_dir}/{episode + 1}"
                )
            ckpt_marker = " [ckpt]"

        print(
            f"[{episode+1:{len(str(max_episodes))}d}/{max_episodes}]"
            f"  acc={cur_accuracy:.4f}  min={min_acc_episode:.4f}"
            f"  best_avg_reward={best_accuracy:.2f}  reward={episode_reward:.2f}"
            f"  greedy={info.get('greedy_accuracy', 0.0):.2f}  path={info.get('path_match', 0.0):.0f}"
            f"{' [push]' if push else ''}{ckpt_marker}"
        )


# ---------------------------------------------------------------------------
# Sacred entry point
# ---------------------------------------------------------------------------

def _parse_kv(text) -> dict:
    """'a:1,b:2.5,c:name' -> {'a': 1, 'b': 2.5, 'c': 'name'}"""
    out = {}
    for part in str(text or '').split(','):
        if ':' not in part:
            continue
        key, value = part.split(':', 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except (TypeError, ValueError):
                pass
        out[key.strip()] = value
    return out


@ex.main
def sacred_main(_run, _config):
    """Main function with Sacred experiment tracking."""
    print(f"Starting Sacred experiment: {_run._id}")
    print(f"Configuration: {_config}")

    victim_run = start_victim_sacred(_config)
    victim_tracker = VictimExperimentTracker(sacred_run=victim_run)
    attack_tracker = AttackExperimentTracker(
        sacred_run=_run, timestep_every=_config.get('log_timestep_every', 20))

    privacy_mode_str = _config.get('privacy_mode', 'full_blackbox')
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
    args.num_victims = _config.get('num_victims', 1)  # default: 1
    args.victim_algo = _config.get('victim_algo', 'qlearning')
    args.env = _config.get('env', 'grid3d')
    args.env_shape = _config.get('env_shape', None)
    args.grid_size = _config.get('grid_size', None)
    args.attack_algo = _config.get('attack_algo', 'ddpg')
    args.keep_checkpoints = _config.get('keep_checkpoints', 3)
    args.encoder_mode = _config.get('encoder_mode', 'lstm')
    args.victim_alpha = _config.get('victim_alpha', 0.1)
    args.attack_rate = _config.get('attack_rate', 0.001)
    args.attack_prate = _config.get('attack_prate', 0.0001)
    args.updates_per_step = _config.get('updates_per_step', 15)
    args.expl_noise = _config.get('expl_noise', 0.1)
    args.reward_mode = _config.get('reward_mode', 'traj')
    args.ae_model_path = _config.get('ae_model_path', None)
    args.lstm_append_dynamics = _config.get('lstm_append_dynamics', True)
    args.td3 = _config.get('td3', False)
    args.normalize_obs = _config.get('normalize_obs', False)
    args.explore_push_prob = _config.get('explore_push_prob', 0.0)
    args.obs_dynamics = _config.get('obs_dynamics', 'actions')
    args.effort_weight = _config.get('effort_weight', 0.0)
    args.victim_kwargs = {**(_config.get('victim_kwargs') or {}), **_parse_kv(_config.get('victim_args', ''))}
    args.lstm_embedding_dim = _config.get('lstm_embedding_dim', None)
    args.lstm_seq_len = _config.get('lstm_seq_len', None)
    args.lstm_memory_size = _config.get('lstm_memory_size', None)
    # Constructor arguments for the environment / victim, given on the command line as
    # "name:value,name:value" (Sacred rejects new keys inside a dict-valued config entry).
    args.env_kwargs = {**(_config.get('env_kwargs') or {}), **_parse_kv(_config.get('env_args', ''))}
    args.push_cem = _config.get('push_cem', False)
    args.cem_batch = _config.get('cem_batch', 16)
    args.cem_elite = _config.get('cem_elite', 4)
    args.cem_lr = _config.get('cem_lr', 0.5)
    args.cem_floor = _config.get('cem_floor', 0.05)
    args.bc_weight = _config.get('bc_weight', 0.0)
    args.elite_k = _config.get('elite_k', 20)

    # Settings a checkpoint needs to be reloaded later (read by eval_attack.py)
    with open(os.path.join(model_dir, 'run_config.json'), 'w') as fh:
        json.dump({k: _config.get(k) for k in (
            'env', 'grid_size', 'env_shape', 'env_kwargs', 'victim_algo', 'encoder_mode',
            'ae_model_path', 'obs_dynamics', 'normalize_obs', 'td3', 'lstm_append_dynamics',
            'victim_n_episodes', 'max_timesteps', 'reward_mode', 'seed', 'effort_weight',
            'victim_kwargs', 'lstm_embedding_dim', 'lstm_seq_len', 'lstm_memory_size')}
                   | {'env_kwargs': args.env_kwargs, 'victim_kwargs': args.victim_kwargs}, fh, indent=1)

    population = build_population(args, privacy_mode, victim_tracker)
    obs_encoder, action_translator = build_encoders(population, args)
    attack_env = build_attack_env(population, args, encoder=obs_encoder)

    state_dim = attack_env.observation_space.shape[0]
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
