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

The run is rebuilt from the run_config.json that main.py wrote next to its checkpoints.

    python score_attack.py --run_dir results/<run> [--episodes 20]

Prints one SCORE line and writes results/<run>/score.json.
"""

import argparse
import json
import os

import numpy as np
import torch

from builders import (
    build_attack_algo,
    build_attack_env,
    build_encoders,
    build_population,
    resolve_config,
)

METRICS = ("greedy_accuracy", "path_match", "target_similarity", "victim_return")


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


def score(run_dir: str, checkpoint: str = "best_model", episodes: int = 20) -> dict:
    cfg, attack_env, translator, attacker, used = load_run(run_dir, checkpoint)
    zeros = np.zeros(attack_env.action_space.shape)
    baseline = {k: [] for k in METRICS}
    attacked = {k: [] for k in METRICS}
    for ep in range(episodes):
        for store, policy in (
            (baseline, lambda obs: zeros),
            (attacked, attacker.act_greedy),
        ):
            np.random.seed(5000 + ep)  # same victim randomness for both conditions
            torch.manual_seed(5000 + ep)
            info = rollout(attack_env, translator, policy, cfg["max_timesteps"])
            for k in METRICS:
                store[k].append(float(info.get(k, 0.0)))
    return {
        "run_dir": run_dir,
        "checkpoint": used,
        "episodes": episodes,
        "env": cfg["env"],
        "env_kwargs": cfg["env_kwargs"],
        "victim_algo": cfg["victim_algo"],
        "baseline": {k: float(np.mean(baseline[k])) for k in METRICS},
        "attacked": {k: float(np.mean(attacked[k])) for k in METRICS},
        "baseline_route_count": int(np.sum(baseline["path_match"])),
        "route_count": int(np.sum(attacked["path_match"])),
        "per_victim": {"baseline": baseline, "attacked": attacked},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best_model")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()

    out = score(a.run_dir, a.checkpoint, a.episodes)
    with open(os.path.join(a.run_dir, "score.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    b, t, n = out["baseline"], out["attacked"], out["episodes"]
    print(
        f"SCORE env={out['env']} victim={out['victim_algo']}"
        f" | target_match {b['greedy_accuracy']:.2f} -> {t['greedy_accuracy']:.2f}"
        f" | route {out['baseline_route_count']}/{n} -> {out['route_count']}/{n}"
        f" | similarity {b['target_similarity']:.2f} -> {t['target_similarity']:.2f}"
        f" | victim_return {b['victim_return']:.1f} -> {t['victim_return']:.1f}"
    )


if __name__ == "__main__":
    main()
