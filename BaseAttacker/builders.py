"""Build the pieces of an attack experiment from a settings dictionary.

    cfg = resolve_config({"env": "ring", "victim_algo": "sarsa"})
    population = build_population(cfg)
    obs_encoder, translator = build_encoders(population, cfg)
    attack_env = build_attack_env(population, cfg, obs_encoder)
    attacker = build_attack_algo(attack_env, cfg)

Nothing here depends on Sacred, so the same functions serve training (main.py),
scoring (score_attack.py) and your own scripts.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
from yacs.config import CfgNode as CN

from ae.action_translator import ActionTranslator
from ae.ae import AEObservationEncoder
from ae.lstm_encoder import LSTMTrajectoryEncoder
from ae.observation_encoder import ObservationEncoder
from ae.victim_encoder import VictimEncoder
from agent.attack_dispatch import AttackDispatch
from agent.victim_system import PrivacyMode, VictimSystem
from defaults import DEFAULT_CONFIG
from envs.attack_environment import AttackEnvironment
from registry import ATTACK_ALGO_MAP, ENV_MAP, VICTIM_ALGO_MAP

_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "config_default.yaml"
)
with open(_YAML) as _fh:
    algo_config = CN.load_cfg(_fh)
algo_config.freeze()


def parse_kv(text) -> dict:
    """'a:1,b:2.5,c:name' -> {'a': 1, 'b': 2.5, 'c': 'name'}"""
    out = {}
    for part in str(text or "").split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except (TypeError, ValueError):
                pass
        out[key.strip()] = value
    return out


def resolve_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Defaults + overrides, with env_args / victim_args folded into the kwargs dicts."""
    cfg = {
        **DEFAULT_CONFIG,
        **{k: v for k, v in (overrides or {}).items() if k in DEFAULT_CONFIG},
    }
    cfg["env_kwargs"] = {**(cfg["env_kwargs"] or {}), **parse_kv(cfg["env_args"])}
    cfg["victim_kwargs"] = {
        **(cfg["victim_kwargs"] or {}),
        **parse_kv(cfg["victim_args"]),
    }
    cfg["env_args"] = cfg["victim_args"] = ""
    if cfg["grid_size"]:
        cfg["env_kwargs"]["grid_dimensions"] = (
            int(cfg["grid_size"]),
            int(cfg["grid_size"]),
        )
    elif cfg["env_shape"]:
        cfg["env_kwargs"]["grid_dimensions"] = tuple(int(v) for v in cfg["env_shape"])
    return cfg


def build_population(cfg: Dict[str, Any], victim_tracker=None) -> VictimSystem:
    """The victims: K copies of (environment, learning algorithm)."""
    env_class = ENV_MAP[cfg["env"]]
    algo_class = VICTIM_ALGO_MAP[cfg["victim_algo"]]
    env_kwargs = dict(cfg["env_kwargs"] or {})
    if "grid_dimensions" in env_kwargs:
        env_kwargs["grid_dimensions"] = tuple(env_kwargs["grid_dimensions"])

    if hasattr(algo_class, "from_env"):
        # Function-approximation victims size themselves from the environment
        algo_kwargs = dict(cfg["victim_kwargs"] or {})
    else:
        algo_kwargs = dict(algo_config.VICTIM.DEFAULT_KWARGS)
        algo_kwargs["alpha"] = cfg["victim_alpha"]
        algo_kwargs.update(cfg["victim_kwargs"] or {})

    privacy = (
        PrivacyMode.FULL_WHITEBOX
        if cfg["privacy_mode"] == "full_whitebox"
        else PrivacyMode.FULL_BLACKBOX
    )
    return VictimSystem(
        num_victims=cfg["num_victims"],
        env_class=env_class,
        env_kwargs=env_kwargs,
        algo_class=algo_class,
        algo_kwargs=algo_kwargs,
        privacy_mode=privacy,
        base_seed=cfg["seed"],
        victim_tracker=victim_tracker,
    )


def build_encoders(population: VictimSystem, cfg: Dict[str, Any]):
    """The attacker's eyes and hands: (ObservationEncoder, ActionTranslator)."""
    env = population.env
    mode = cfg["encoder_mode"]

    # Context = the non-behaviour part of the observation
    if cfg["obs_dynamics"] == "env":
        context_dim = env.dynamics_dim
        bounds = env.dynamics_bounds() if cfg["normalize_obs"] else None
        initial_context = env.get_dynamics() if cfg["normalize_obs"] else None
    elif cfg["obs_dynamics"] == "actions":
        context_dim = int(np.prod(env.perturbation_space.shape))
        bounds, initial_context = None, None  # the running sum starts at zero
    else:
        context_dim, bounds, initial_context = 0, None, None

    if mode == "ae":
        inner = AEObservationEncoder.from_population(
            population,
            model_path=cfg["ae_model_path"],
            online_episodes=cfg["ae_online_episodes"],
            online_epochs=cfg["ae_online_epochs"],
            context_dim=context_dim,
            context_bounds=bounds,
            initial_context=initial_context,
        )
    elif mode == "lstm":
        inner = LSTMTrajectoryEncoder.from_population(
            population,
            algo_config,
            embedding_dim=cfg["lstm_embedding_dim"],
            seq_len=cfg["lstm_seq_len"],
            memory_size=cfg["lstm_memory_size"],
        )
        trajectories = env.target_trajectories()
        if trajectories is not None:
            inner.set_target_trajectories(trajectories)  # target = recorded behaviour
        else:
            inner.populate_target_memory(
                env, population.target, n_steps=inner.seq_len * 2
            )
    elif mode == "whitebox":
        inner = VictimEncoder.from_population(population)
    else:
        raise ValueError(f"encoder_mode must be ae|lstm|whitebox, got {mode!r}")

    # The AE encoder appends the context itself; the LSTM encoder gets it appended here.
    append = mode == "lstm" and cfg["lstm_append_dynamics"]
    obs_encoder = ObservationEncoder(
        inner,
        append_dynamics_dim=context_dim if append else 0,
        normalize_dynamics=bounds is not None,
        dynamics_bounds=bounds,
        initial_dynamics=initial_context if append else None,
    )
    return obs_encoder, ActionTranslator.from_space(env.perturbation_space)


def build_attack_env(
    population: VictimSystem, cfg: Dict[str, Any], encoder=None
) -> AttackEnvironment:
    """The attacker's environment: one step = perturb, let the victims relearn, observe."""
    return AttackEnvironment(
        victim_population=population,
        attack_dispatch=AttackDispatch(),
        victim_train_episodes=cfg["victim_n_episodes"],
        encoder=encoder,
        reward_mode=cfg["reward_mode"],
        hold_threshold=cfg["hold_threshold"],
        obs_dynamics=cfg["obs_dynamics"],
        effort_weight=cfg["effort_weight"],
    )


def build_attack_algo(attack_env: AttackEnvironment, cfg: Dict[str, Any]):
    """The attacker's learner, sized from the attack environment's spaces."""
    state_dim = attack_env.observation_space.shape[0]
    action_dim = attack_env.action_space.shape[0]
    max_action = float(np.max(np.abs(attack_env.action_space.high)))
    algo_class = ATTACK_ALGO_MAP[cfg["attack_algo"]]

    if cfg["attack_algo"] == "ppo":
        kwargs = dict(algo_config.ATTACK.PPO_KWARGS)
    else:
        kwargs = dict(algo_config.ATTACK.DEFAULT_KWARGS)
        kwargs.update(
            tau=cfg["tau"],
            discount=cfg["discount"],
            eps_greedy_start_episodes=cfg["eps_greedy_start_episodes"],
            rate=cfg["attack_rate"],
            prate=cfg["attack_prate"],
            updates_per_step=cfg["updates_per_step"],
            bsize=cfg["batch_size"],
            expl_noise=cfg["expl_noise"],
            td3=cfg["td3"],
            bc_weight=cfg["bc_weight"],
            correction_scale=cfg["push_correction"],
        )
    kwargs.update(nb_states=state_dim, nb_actions=action_dim, max_action=max_action)
    return algo_class(**kwargs)
