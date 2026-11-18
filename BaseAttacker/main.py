"""Train an environment-poisoning attacker.

    python main.py                                   # defaults: 4x4 grid, Q-learning victim
    python main.py with quick_test                   # finishes in seconds
    python main.py with env=ring victim_algo=sarsa max_episodes=3000 seed=1
    python main.py with lander                       # LunarLander, DQN victim

All settings and their defaults are in defaults.py. Each run writes its checkpoints
and a run_config.json to `model_dir`; score it afterwards with score_attack.py.

The loop, per attack episode:
    1. fresh victims, unperturbed environment
    2. repeat `max_timesteps` times:
         attacker observes -> sends a perturbation -> victims relearn -> attacker is
         rewarded from the victims' observed behaviour
    3. the attacker updates from its replay buffer
"""

import json
import os
import time

import numpy as np
import torch

from agent.trackers import AttackExperimentTracker, VictimExperimentTracker
from attack.push_exploration import PushExploration
from builders import (
    build_attack_algo,
    build_attack_env,
    build_encoders,
    build_population,
    resolve_config,
)
from experiment import ex, finish_victim_sacred, start_victim_sacred
from naming import run_name, run_path
from publish import _git_commit

# Where runs are written. On clusters whose code lives on a small or read-mostly
# filesystem, point this at scratch storage: RESULTS_ROOT=$SCRATCH/envpois/results
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results")

CHECKPOINT_SUFFIXES = (
    "_actor",
    "_actor_optimizer",
    "_critic",
    "_critic_optimizer",
    "_critic2",
    "_critic2_optimizer",
)


def _delete_old_checkpoints(
    model_dir: str, episode: int, every: int, keep: int
) -> None:
    old = episode - keep * every
    if old <= 0:
        return
    for suffix in CHECKPOINT_SUFFIXES:
        try:
            os.remove(os.path.join(model_dir, f"{old}{suffix}"))
        except FileNotFoundError:
            pass


def train(
    attack_env, attack_algo, obs_encoder, action_translator, cfg, attack_tracker=None
):
    """Run the attack training loop described in the module docstring."""
    max_episodes, max_timesteps = cfg["max_episodes"], cfg["max_timesteps"]
    model_dir, every = cfg["model_dir"], cfg["eval_freq_episode"]
    # Exploration-only episodes: the learner's own warm-up, or longer if the behaviour
    # encoder is fitted on the run's first episodes (ae_model_path="online").
    encoder_warmup = obs_encoder.warmup_episodes
    warmup_episodes = max(attack_algo.warmup_episodes, encoder_warmup)
    os.makedirs(model_dir, exist_ok=True)

    victim_env = attack_env.victim_system.env
    victim_env.reset_dynamics()
    initial_dynamics = np.asarray(victim_env.get_dynamics(), dtype=float).copy()

    push = PushExploration(
        attack_env.action_space.shape,
        prob=cfg["explore_push_prob"],
        adaptive=cfg["push_cem"],
        batch=cfg["cem_batch"],
        elite=cfg["cem_elite"],
        lr=cfg["cem_lr"],
        floor=cfg["cem_floor"],
        patience=cfg["cem_patience"],
    )

    best_avg_reward = 0.0  # best 50-episode mean of the episode reward so far
    recent_rewards = []  # rewards of the last 50 non-push episodes
    elites = []  # (episode reward, observations, actions) for imitation

    for episode in range(max_episodes):
        obs, _ = attack_env.reset()
        started = time.time()
        warm = episode < warmup_episodes
        is_push = push.start_episode(warm)
        episode_reward, min_reward, info = 0.0, 1.0, {}
        greedy_sum = route_sum = 0.0
        ep_obs, ep_act = [], []

        for timestep in range(max_timesteps):
            if is_push:
                raw_action = push.action()
            elif warm:
                raw_action = attack_env.action_space.sample()
            else:
                raw_action = attack_algo.act(np.array(obs))

            next_obs, reward, terminated, _, info = attack_env.step(
                action_translator.translate(raw_action)
            )
            attack_algo.store_transition(obs, raw_action, next_obs, reward, terminated)
            ep_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            ep_act.append(np.asarray(raw_action, dtype=np.float32).reshape(-1))

            episode_reward += reward
            min_reward = min(min_reward, reward)
            greedy_sum += info.get("greedy_accuracy", 0.0)
            route_sum += info.get("path_match", 0.0)
            if attack_tracker is not None:
                attack_tracker.log_timestep(episode, timestep, info)
            obs = next_obs
            if terminated:
                break
        n_steps = timestep + 1

        if is_push:
            push.end_episode(episode_reward)
        if cfg["push_base"]:
            attack_algo.set_base(push.base(cfg["push_base_conf"]))

        if encoder_warmup and episode + 1 == warmup_episodes:
            accuracy = obs_encoder.fit()
            attack_algo.clear_buffer()  # those observations used the unfitted encoder
            print(
                f"behaviour encoder fitted on the first {warmup_episodes} episodes: "
                f"held-out reconstruction accuracy {accuracy:.3f}"
            )

        if cfg["bc_weight"] > 0 and (
            len(elites) < cfg["elite_k"] or episode_reward > elites[-1][0]
        ):
            elites = sorted(
                elites + [(episode_reward, ep_obs, ep_act)], key=lambda e: -e[0]
            )[: cfg["elite_k"]]
            attack_algo.set_elite(
                np.concatenate([np.stack(o) for _, o, _ in elites]),
                np.concatenate([np.stack(a) for _, _, a in elites]),
            )

        obs_encoder.train_encoder()  # only encoders that learn online do anything here

        losses = {}
        if not warm and attack_algo.ready_to_train():
            losses = attack_algo.update(episode=episode) or {}

        # Checkpoint selection uses the attacker's own reward only (no ground truth).
        if not is_push:
            recent_rewards = (recent_rewards + [episode_reward])[-50:]
        avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        if episode >= warmup_episodes + 50 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            attack_algo.save(os.path.join(model_dir, "best_model"))
            obs_encoder.save(os.path.join(model_dir, "best_model_encoder.pt"))

        marker = " [push]" if is_push else ""
        if (episode + 1) % every == 0 or episode + 1 == max_episodes:
            attack_algo.save(os.path.join(model_dir, str(episode + 1)))
            obs_encoder.save(os.path.join(model_dir, "latest_encoder.pt"))
            # The exploration search's state, so it can be scored on its own
            # (score_attack.py --policy push)
            np.save(os.path.join(model_dir, "push_probs.npy"), push.probs)
            _delete_old_checkpoints(
                model_dir, episode + 1, every, cfg["keep_checkpoints"]
            )
            if attack_tracker is not None:
                attack_tracker.log_checkpoint(
                    episode + 1, reward, f"{model_dir}/{episode + 1}"
                )
            marker += " [ckpt]"

        if attack_tracker is not None:
            attack_tracker.log_episode(
                episode,
                {
                    "accuracy": reward,  # reward at the last attack step
                    "min_accuracy": min_reward,
                    "episode_reward": episode_reward,
                    "time": time.time() - started,
                    # Ground truth, for evaluation only; never shown to the attacker:
                    "greedy_accuracy": info.get("greedy_accuracy", 0.0),
                    "path_match": info.get("path_match", 0.0),
                    "greedy_accuracy_mean": greedy_sum / n_steps,
                    "path_match_mean": route_sum / n_steps,
                    "dynamics_change": float(
                        np.abs(victim_env.get_dynamics() - initial_dynamics).mean()
                    ),
                    **losses,
                },
                losses.get("critic_loss"),
            )

        print(
            f"[{episode + 1:{len(str(max_episodes))}d}/{max_episodes}]"
            f"  acc={reward:.4f}  min={min_reward:.4f}"
            f"  best_avg_reward={best_avg_reward:.2f}  reward={episode_reward:.2f}"
            f"  greedy={info.get('greedy_accuracy', 0.0):.2f}  path={info.get('path_match', 0.0):.0f}"
            f"{marker}"
        )


@ex.main
def sacred_main(_run, _config):
    cfg = resolve_config(dict(_config))
    if cfg["model_dir"] is None:
        cfg["model_dir"] = os.path.join(RESULTS_ROOT, run_path(cfg))
    cfg["run_name"] = run_name(cfg)
    cfg["code_commit"] = _git_commit()
    os.makedirs(cfg["model_dir"], exist_ok=True)
    _run.info["model_dir"] = cfg["model_dir"]
    # Everything needed to rebuild this run later (read by score_attack.py)
    with open(os.path.join(cfg["model_dir"], "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    victim_run = start_victim_sacred(cfg)
    try:
        population = build_population(
            cfg, VictimExperimentTracker(sacred_run=victim_run)
        )
        obs_encoder, action_translator = build_encoders(population, cfg)
        attack_env = build_attack_env(population, cfg, obs_encoder)
        # The space has its own generator (used by the warm-up episodes); unseeded, it
        # makes two runs with the same seed differ from the first episode on.
        attack_env.action_space.seed(cfg["seed"])
        attack_algo = build_attack_algo(attack_env, cfg)
        attack_tracker = AttackExperimentTracker(
            sacred_run=_run, timestep_every=cfg["log_timestep_every"]
        )
        train(
            attack_env, attack_algo, obs_encoder, action_translator, cfg, attack_tracker
        )
        finish_victim_sacred(victim_run, status="COMPLETED")
    except BaseException as exc:
        # Always close the victim run: its heartbeat thread would keep the process alive.
        finish_victim_sacred(victim_run, status=f"FAILED: {exc}")
        raise


@ex.command(unobserved=True)
def print_run_dir(_config):
    """Print the folder a run with these settings writes to (used by the sweep scripts)."""
    cfg = resolve_config(dict(_config))
    print(cfg["model_dir"] or os.path.join(RESULTS_ROOT, run_path(cfg)))


if __name__ == "__main__":
    ex.run_commandline()
