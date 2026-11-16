"""Score a trained attacker on fresh victims. Works for every environment.

For each of N fresh victims two attack episodes are run with the same seed:
  baseline  the attacker sends zeros (no poisoning)
  attacked  the trained attacker acts, without exploration

and the ground-truth metrics at the final attack step are compared:
  target_match  share of target states where the victim's greedy action is the target's
                (per-state table for grid/ring; recorded target states for the lander)
  route         1 if the victim's greedy route is exactly the target route (grid/ring)
  similarity    behaviour-embedding similarity to the target (lander)

Everything needed to rebuild the run comes from run_config.json written by main.py.

    python score_attack.py --run_dir results/<run> [--episodes 20]

Writes results/<run>/score.json and prints one summary line.
"""

import argparse
import json
import os

os.environ.setdefault(
    "SACRED_FILE_ONLY", "1"
)  # importing main must not open a Mongo run

import numpy as np
import torch

from agent.victim_system import PrivacyMode
from main import build_attack_algo, build_attack_env, build_encoders, build_population


def rollout(env, translator, policy, steps):
    """One attack episode; returns the info dict of the last step."""
    obs, _ = env.reset()
    info = {}
    for _ in range(steps):
        obs, _, _, _, info = env.step(
            translator.translate(policy(np.asarray(obs, dtype=np.float32)))
        )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best_model")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run_dir, "run_config.json")))

    class Args:
        pass

    args = Args()
    args.__dict__.update(cfg)
    args.__dict__.update(
        num_victims=1, attack_algo="ddpg", expl_noise=0.0, victim_alpha=0.1
    )
    steps = int(cfg.get("max_timesteps") or 15)

    pop = build_population(args, PrivacyMode.FULL_BLACKBOX)
    obs_enc, translator = build_encoders(pop, args)
    env = build_attack_env(pop, args, encoder=obs_enc)
    algo = build_attack_algo(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
        args,
    )
    ckpt = os.path.join(a.run_dir, a.checkpoint)
    if not os.path.exists(ckpt + "_actor"):
        # No best checkpoint was saved (run too short): fall back to the latest numbered one
        nums = sorted(
            int(f.split("_")[0])
            for f in os.listdir(a.run_dir)
            if f.endswith("_actor") and f.split("_")[0].isdigit()
        )
        ckpt = os.path.join(a.run_dir, str(nums[-1]))
        enc_path = os.path.join(a.run_dir, "latest_encoder.pt")
    else:
        enc_path = ckpt + "_encoder.pt"
    algo.load(ckpt)
    algo.eval()
    obs_enc.load(enc_path)

    zeros = np.zeros(env.action_space.shape)
    keys = ("greedy_accuracy", "path_match", "target_similarity", "victim_return")
    base = {k: [] for k in keys}
    att = {k: [] for k in keys}
    for ep in range(a.episodes):
        for store, policy in (
            (base, lambda o: zeros),
            (att, algo.select_on_policy_action),
        ):
            np.random.seed(5000 + ep)
            torch.manual_seed(5000 + ep)
            info = rollout(env, translator, policy, steps)
            for k in keys:
                store[k].append(float(info.get(k, 0.0)))

    def mean(d, k):
        return float(np.mean(d[k]))

    out = {
        "run_dir": a.run_dir,
        "checkpoint": os.path.basename(ckpt),
        "episodes": a.episodes,
        "env": cfg.get("env"),
        "env_kwargs": cfg.get("env_kwargs"),
        "grid_size": cfg.get("grid_size"),
        "victim_algo": cfg.get("victim_algo"),
        "baseline": {k: mean(base, k) for k in keys},
        "attacked": {k: mean(att, k) for k in keys},
        "route_count": int(np.sum(att["path_match"])),
        "baseline_route_count": int(np.sum(base["path_match"])),
    }
    json.dump(out, open(os.path.join(a.run_dir, "score.json"), "w"), indent=1)
    print(
        "SCORE env={env} victim={victim_algo} | target_match {b:.2f} -> {t:.2f} | route {br}/{n} -> {r}/{n} "
        "| similarity {bs:.2f} -> {s:.2f} | victim_return {bv:.1f} -> {v:.1f}".format(
            env=out["env"],
            victim_algo=out["victim_algo"],
            n=a.episodes,
            b=out["baseline"]["greedy_accuracy"],
            t=out["attacked"]["greedy_accuracy"],
            br=out["baseline_route_count"],
            r=out["route_count"],
            bs=out["baseline"]["target_similarity"],
            s=out["attacked"]["target_similarity"],
            bv=out["baseline"]["victim_return"],
            v=out["attacked"]["victim_return"],
        )
    )


if __name__ == "__main__":
    main()
