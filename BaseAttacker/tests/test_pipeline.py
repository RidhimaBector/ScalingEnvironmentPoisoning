"""End-to-end: build, train for a few episodes, save, reload and score.

These are smoke tests of the plumbing with tiny budgets; they say nothing about
whether the attack succeeds.
"""
import json
import os

import numpy as np
import pytest

from builders import (build_attack_algo, build_attack_env, build_encoders, build_population,
                      resolve_config)
from main import train
from score_attack import score

TINY = dict(max_episodes=6, max_timesteps=3, victim_n_episodes=5, eps_greedy_start_episodes=2,
            batch_size=4, eval_freq_episode=3)


def run(tmp_path, **settings):
    cfg = resolve_config({**TINY, **settings, "model_dir": str(tmp_path)})
    with open(os.path.join(cfg["model_dir"], "run_config.json"), "w") as fh:
        json.dump(cfg, fh)
    np.random.seed(0)
    try:
        population = build_population(cfg)
    except ImportError as exc:
        pytest.skip(str(exc))
    obs_encoder, translator = build_encoders(population, cfg)
    attack_env = build_attack_env(population, cfg, obs_encoder)
    attacker = build_attack_algo(attack_env, cfg)
    train(attack_env, attacker, obs_encoder, translator, cfg)
    return cfg, attack_env


@pytest.mark.parametrize("settings", [
    dict(env="grid3d", victim_algo="qlearning"),
    dict(env="grid3d", victim_algo="sarsa", obs_dynamics="env", normalize_obs=True),
    dict(env="grid3d", victim_algo="reinforce", reward_mode="mix", td3=True, bc_weight=1.0),
    dict(env="grid3d", victim_algo="qlearning", encoder_mode="lstm", obs_dynamics="none"),
    dict(env="grid3d", victim_algo="qlearning", encoder_mode="whitebox", privacy_mode="full_whitebox"),
    dict(env="ring", victim_algo="qlearning"),
    dict(env="ring", victim_algo="sarsa", explore_push_prob=0.0),
    dict(env="grid3d", victim_algo="qlearning", push_base=True, explore_push_prob=0.5),
    dict(env="ring", victim_algo="sarsa", ae_model_path="online", ae_online_episodes=4,
         ae_online_epochs=2, push_base=True),
    dict(env="lunarlander", victim_algo="dqn", encoder_mode="lstm", reward_mode="embedding",
         effort_weight=0.1, lstm_seq_len=20, lstm_memory_size=200),
], ids=lambda s: "-".join(str(v) for v in s.values()))
def test_train_then_score(tmp_path, settings):
    if settings.get("privacy_mode") == "full_whitebox":
        pytest.importorskip("ot", reason="white-box distance metrics need the optional POT package")
    cfg, attack_env = run(tmp_path, **settings)
    assert os.path.exists(tmp_path / "6_actor"), "the final checkpoint must always be saved"

    result = score(str(tmp_path), episodes=2)
    assert result["episodes"] == 2
    for condition in ("baseline", "attacked"):
        assert 0.0 <= result[condition]["greedy_accuracy"] <= 1.0


def test_attacker_only_sees_its_own_actions(tmp_path):
    """obs_dynamics="actions": the context is the running sum of what the attacker sent,
    not the environment's dynamics."""
    cfg, attack_env = run(tmp_path, env="grid3d", victim_algo="qlearning")
    attack_env.reset()
    step = np.full(attack_env.action_space.shape, 0.5)
    obs, *_ = attack_env.step(step)
    obs, *_ = attack_env.step(step)
    n = int(np.prod(attack_env.action_space.shape))
    assert np.allclose(obs[-n:], 1.0)             # 0.5 + 0.5, although the altitude is ~6-9
