"""Score a trained attacker on fresh victims. Works for every environment.

For each of N fresh victims two attack episodes are run with the same seed:

  baseline   the attacker sends zeros (no poisoning)
  attacked   the trained attacker acts, without exploration

and ground truth at the final attack step is compared:

  target_match   share of target states where the victim's greedy action is the
                 target's (policy table for tabular victims; recorded target states
                 for victims with continuous states)
  route          1 if the victim's greedy route is exactly the target route
                 (environments that define a route)
  similarity     similarity of the victim's behaviour embedding to the target's
                 (reward_mode="embedding" runs)
  victim_return  the victim's own mean episode return
  state_gap      targets given as recorded trajectories: distance between the mean
                 state the victim visited in the final step and the target's mean
                 state, in units of the target's per-dimension spread (lower = closer)
  action_gap     same targets: total-variation distance between how often the victim
                 took each action and how often the target did (0 = same, 1 = disjoint)

The run is rebuilt from the run_config.json that main.py wrote next to its checkpoints.

    python score_attack.py --run_dir results/<run> [--episodes 20]

Prints one SCORE line and writes results/<run>/score.json.

--policy chooses what plays the "attacked" episode:

  actor        the trained attacker (default)
  push         the exploration search's most likely direction per dimension, the same
               at every step, no noise (from push_probs.npy; writes score_push.json)
  push_sample  directions drawn from the search's probabilities once per victim, plus
               the exploration noise at every step, as in a training push episode
               (writes score_push_sample.json)
"""

import argparse
import json
import os

import numpy as np
import torch

from attack.push_exploration import base_from_probs
from builders import (
    build_attack_algo,
    build_attack_env,
    build_encoders,
    build_population,
    resolve_config,
)

METRICS = ("greedy_accuracy", "path_match", "target_similarity", "victim_return")
GAPS = ("state_gap", "action_gap")


def trajectory_gaps(victim_results, recorded) -> dict:
    """How far the victim's observed states and actions are from a recorded target."""
    steps = [t for r in victim_results for t in r.get("trajectories", [])]
    if not steps or not recorded:
        return {}
    states = np.stack([np.asarray(t[2], dtype=float).reshape(-1) for t in steps])
    target_states = np.stack([np.asarray(t[2], dtype=float).reshape(-1) for t in recorded])
    spread = target_states.std(axis=0) + 1e-8
    state_gap = np.linalg.norm((states.mean(axis=0) - target_states.mean(axis=0)) / spread)
    actions = np.array([int(t[3]) for t in steps])
    target_actions = np.array([int(t[3]) for t in recorded])
    n = int(max(actions.max(), target_actions.max())) + 1
    freq = np.bincount(actions, minlength=n) / len(actions)
    target_freq = np.bincount(target_actions, minlength=n) / len(target_actions)
    return {
        "state_gap": float(state_gap / np.sqrt(states.shape[1])),
        "action_gap": float(0.5 * np.abs(freq - target_freq).sum()),
    }


def load_run(run_dir: str, checkpoint: str = "best_model", overrides=None):
    """Rebuild a trained run: returns (cfg, attack_env, translator, attacker, checkpoint name)."""
    with open(os.path.join(run_dir, "run_config.json")) as fh:
        cfg = resolve_config({**json.load(fh), **(overrides or {})})
    population = build_population(cfg)
    obs_encoder, translator = build_encoders(population, cfg)
    attack_env = build_attack_env(population, cfg, obs_encoder)
    attacker = build_attack_algo(attack_env, cfg)

    path = os.path.join(run_dir, checkpoint)
    encoder_path = path + "_encoder.pt"
    if not os.path.exists(path + "_actor"):
        # No "best" checkpoint yet (very short run): use the latest numbered one.
        numbered = sorted(
            int(f.split("_")[0])
            for f in os.listdir(run_dir)
            if f.endswith("_actor") and f.split("_")[0].isdigit()
        )
        if not numbered:
            raise FileNotFoundError(f"no checkpoint found in {run_dir}")
        path = os.path.join(run_dir, str(numbered[-1]))
        encoder_path = os.path.join(run_dir, "latest_encoder.pt")
    attacker.load(path)
    probs_path = os.path.join(run_dir, "push_probs.npy")
    if cfg["push_base"] and cfg["push_final_base"] and os.path.exists(probs_path):
        # Use the search's final result as the base, not the one saved with the
        # checkpoint (the best checkpoint can predate the search's last changes).
        attacker.set_base(base_from_probs(np.load(probs_path), cfg["push_base_conf"]))
    attacker.eval()
    obs_encoder.load(encoder_path)  # only encoders trained online have weights to load
    return cfg, attack_env, translator, attacker, os.path.basename(path)


def rollout(attack_env, translator, policy, steps: int) -> dict:
    """One attack episode; returns the info dict of the last step."""
    obs, _ = attack_env.reset()
    info = {}
    for _ in range(steps):
        obs, _, _, _, info = attack_env.step(
            translator.translate(policy(np.asarray(obs, dtype=np.float32)))
        )
    return info


PUSH_NOISE = 0.2  # PushExploration's default, which main.py uses


def push_policy(probs: np.ndarray, shape, sample: bool, seed: int):
    """A fixed-direction policy from the exploration search's probabilities.

    Uses its own random generator, so the victims' randomness stays the same as in the
    baseline episode."""
    rng = np.random.default_rng(seed)
    if sample:
        idx = np.array([rng.choice(3, p=p) for p in probs])
    else:
        idx = probs.argmax(axis=1)
    direction = idx.astype(float) - 1.0

    def policy(obs):
        noise = rng.normal(0.0, PUSH_NOISE, size=direction.shape) if sample else 0.0
        return np.clip(direction + noise, -1.0, 1.0).reshape(shape)

    return policy


def score(
    run_dir: str,
    checkpoint: str = "best_model",
    episodes: int = 20,
    policy_name: str = "actor",
) -> dict:
    cfg, attack_env, translator, attacker, used = load_run(run_dir, checkpoint)
    zeros = np.zeros(attack_env.action_space.shape)
    probs = None
    if policy_name != "actor":
        probs = np.load(os.path.join(run_dir, "push_probs.npy"))
        used = "push_probs.npy"
    recorded = attack_env.victim_system.env.target_trajectories()
    names = METRICS + (GAPS if recorded else ())
    baseline = {k: [] for k in names}
    attacked = {k: [] for k in names}
    for ep in range(episodes):
        for store, policy in (
            (baseline, lambda obs: zeros),
            (
                attacked,
                attacker.act_greedy
                if probs is None
                else push_policy(
                    probs,
                    attack_env.action_space.shape,
                    sample=policy_name == "push_sample",
                    seed=5000 + ep,
                ),
            ),
        ):
            np.random.seed(5000 + ep)  # same victim randomness for both conditions
            torch.manual_seed(5000 + ep)
            info = rollout(attack_env, translator, policy, cfg["max_timesteps"])
            if recorded:
                info = {**info, **trajectory_gaps(info.get("victim_results", []), recorded)}
            for k in names:
                store[k].append(float(info.get(k, 0.0)))
    return {
        "run_dir": run_dir,
        "checkpoint": used,
        "policy": policy_name,
        "episodes": episodes,
        "env": cfg["env"],
        "env_kwargs": cfg["env_kwargs"],
        "victim_algo": cfg["victim_algo"],
        "baseline": {k: float(np.mean(baseline[k])) for k in names},
        "attacked": {k: float(np.mean(attacked[k])) for k in names},
        "baseline_route_count": int(np.sum(baseline["path_match"])),
        "route_count": int(np.sum(attacked["path_match"])),
        "per_victim": {"baseline": baseline, "attacked": attacked},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best_model")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument(
        "--policy", default="actor", choices=("actor", "push", "push_sample")
    )
    a = ap.parse_args()

    out = score(a.run_dir, a.checkpoint, a.episodes, a.policy)
    name = "score.json" if a.policy == "actor" else f"score_{a.policy}.json"
    with open(os.path.join(a.run_dir, name), "w") as fh:
        json.dump(out, fh, indent=1)
    b, t, n = out["baseline"], out["attacked"], out["episodes"]
    print(
        f"SCORE env={out['env']} victim={out['victim_algo']}"
        + ("" if a.policy == "actor" else f" policy={a.policy}")
        +        f" | target_match {b['greedy_accuracy']:.2f} -> {t['greedy_accuracy']:.2f}"
        f" | route {out['baseline_route_count']}/{n} -> {out['route_count']}/{n}"
        f" | similarity {b['target_similarity']:.2f} -> {t['target_similarity']:.2f}"
        f" | victim_return {b['victim_return']:.1f} -> {t['victim_return']:.1f}"
        + (
            f" | state_gap {b['state_gap']:.2f} -> {t['state_gap']:.2f}"
            f" | action_gap {b['action_gap']:.2f} -> {t['action_gap']:.2f}"
            if "state_gap" in b
            else ""
        )
    )


if __name__ == "__main__":
    main()
