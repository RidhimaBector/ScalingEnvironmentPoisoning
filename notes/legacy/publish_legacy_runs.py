"""One-off: publish the November 2026 runs on fir, which predate the naming convention.

Those runs live in results/fir_<old name>_s<seed> in the pre-cleanup checkout. This
script gives each a name in the current convention (naming.py), writes the
run_config.json / score.json the publisher expects when they are missing, and uploads
through publish.py. Scores of the early rounds come from the evaluation lines in the
SLURM logs (20 fresh victims, exact target route at the last attack step).

Run from the old checkout's BaseAttacker/ on fir, with the release code on PYTHONPATH:

    cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker
    PYTHONPATH=~/scratch/thesis_sep/release/BaseAttacker python \\
        ~/scratch/thesis_sep/release/notes/legacy/publish_legacy_runs.py [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import sys

GRID4 = dict(env="grid3d", env_kwargs={"grid_dimensions": [4, 4]}, victim_algo="qlearning")
NEW_AE = "retrained trace autoencoder"

# old-name prefix -> (experiment, config_name, settings that defined it, max_episodes)
FAMILIES = [
    (r"fir_aeold_trace", "r01-encoder-and-reward", "original-ae-trace", dict(GRID4, reward_mode="trace", encoder="original trace autoencoder"), 3000),
    (r"fir_aenew_trace", "r01-encoder-and-reward", "retrained-ae-trace", dict(GRID4, reward_mode="trace", encoder=NEW_AE), 3000),
    (r"fir_aenew_traj", "r01-encoder-and-reward", "retrained-ae-traj", dict(GRID4, reward_mode="traj", encoder=NEW_AE), 3000),
    (r"fir_lstmdyn_trace", "r01-encoder-and-reward", "lstm-trace", dict(GRID4, reward_mode="trace", encoder_mode="lstm"), 3000),
    (r"fir_v2_ae_trace", "r02-stabilised-bundle", "ae-trace", dict(GRID4, reward_mode="trace", td3=True, normalize_obs=True, explore_push_prob=0.1), 5000),
    (r"fir_v2_ae_mix", "r02-stabilised-bundle", "ae-mix", dict(GRID4, reward_mode="mix", td3=True, normalize_obs=True, explore_push_prob=0.1), 5000),
    (r"fir_v2_ae_traj", "r02-stabilised-bundle", "ae-traj", dict(GRID4, reward_mode="traj", td3=True, normalize_obs=True, explore_push_prob=0.1), 5000),
    (r"fir_v2_lstm_mix", "r02-stabilised-bundle", "lstm-mix", dict(GRID4, reward_mode="mix", encoder_mode="lstm", td3=True, normalize_obs=True, explore_push_prob=0.1), 5000),
    (r"fir_v3_base", "r03-ablation-4x4", "base", dict(GRID4, reward_mode="trace"), 3000),
    (r"fir_v3_td3", "r03-ablation-4x4", "td3", dict(GRID4, reward_mode="trace", td3=True), 3000),
    (r"fir_v3_norm", "r03-ablation-4x4", "normalised-input", dict(GRID4, reward_mode="trace", normalize_obs=True), 3000),
    (r"fir_v3_push", "r03-ablation-4x4", "uniform-push", dict(GRID4, reward_mode="trace", explore_push_prob=0.1), 3000),
    (r"fir_v3_mix", "r03-ablation-4x4", "mix-reward", dict(GRID4, reward_mode="mix"), 3000),
    (r"fir_v3_noise", "r03-ablation-4x4", "noise-0.2", dict(GRID4, reward_mode="trace", expl_noise=0.2), 3000),
    (r"fir_v4_push", "r04-uniform-push", "uniform-push", dict(GRID4, reward_mode="trace", explore_push_prob=0.1), 3000),
    (r"fir_v5_cembc1", "r05-adaptive-push-imitation", "adaptive-push-imitation-1", dict(GRID4, explore_push_prob=0.2, push_cem=True, bc_weight=1), 3000),
    (r"fir_v5_cembc", "r05-adaptive-push-imitation", "adaptive-push-imitation-5", dict(GRID4, explore_push_prob=0.2, push_cem=True, bc_weight=5), 3000),
    (r"fir_v5_cem", "r05-adaptive-push-imitation", "adaptive-push", dict(GRID4, explore_push_prob=0.2, push_cem=True), 3000),
    (r"fir_v5_bc", "r05-adaptive-push-imitation", "uniform-push-imitation-5", dict(GRID4, explore_push_prob=0.1, bc_weight=5), 3000),
    (r"fir_v6_cembc1", "r06-adaptive-push-6000", "adaptive-push-imitation-1", dict(GRID4, explore_push_prob=0.2, push_cem=True, bc_weight=1), 6000),
    (r"fir_v6_cem", "r06-adaptive-push-6000", "adaptive-push", dict(GRID4, explore_push_prob=0.2, push_cem=True), 6000),
    (r"fir_grid(?P<n>\d)_cem", "r08-larger-grids-first", "adaptive-push", dict(explore_push_prob=0.2, push_cem=True), 8000),
    (r"fir_tune(?P<n>\d)_floor", "r09-grid-size-tuning", "push20", dict(explore_push_prob=0.2, push_cem=True, cem_floor=-1), 12000),
    (r"fir_tune(?P<n>\d)_push40", "r09-grid-size-tuning", "push40", dict(explore_push_prob=0.4, push_cem=True, cem_floor=-1), 12000),
    (r"fir_tune(?P<n>\d)_slow", "r09-grid-size-tuning", "push40-batch32", dict(explore_push_prob=0.4, push_cem=True, cem_floor=-1, cem_batch=32, cem_elite=8, cem_lr=0.3), 12000),
    (r"fir_agn_grid3d_(?P<victim>\w+?)", "r10-own-actions-observation", "default", dict(GRID4, obs_dynamics="actions", explore_push_prob=0.2, push_cem=True), 6000),
    (r"fir_agn_ring_(?P<victim>\w+?)", "r10-own-actions-observation", "default", dict(env="ring", env_kwargs={"n_states": 8}, obs_dynamics="actions", explore_push_prob=0.2, push_cem=True), 3000),
    (r"fir_s27_grid(?P<n>\d)_(?P<victim>\w+?)", "sweep-3x3x3", "default", dict(obs_dynamics="actions", push_cem=True, cem_floor=-1), None),
    (r"fir_s27_ring(?P<states>\d+)_(?P<victim>\w+?)", "sweep-3x3x3", "default", dict(env="ring", obs_dynamics="actions", push_cem=True, cem_floor=-1), 3000),
    (r"fir_s27_lander(?P<kick>\d+)_(?P<victim>\w+?)", "sweep-3x3x3", "default", dict(env="lunarlander", obs_dynamics="actions", encoder_mode="lstm", reward_mode="embedding"), 1000),
]


def describe(old_name):
    """(experiment, config dict in the current convention) for an old results folder name."""
    for pattern, experiment, config_name, settings, episodes in FAMILIES:
        m = re.fullmatch(pattern + r"_s(?P<seed>\d+)", old_name)
        if not m:
            continue
        g = m.groupdict()
        cfg = dict(env="grid3d", victim_algo="qlearning", env_kwargs={}, obs_dynamics="env",
                   encoder_mode="ae", reward_mode="trace")
        cfg.update(settings)
        if g.get("n"):
            cfg.update(env="grid3d", env_kwargs={"grid_dimensions": [int(g["n"]), int(g["n"])]})
        if g.get("states"):
            cfg["env_kwargs"] = {"n_states": int(g["states"])}
        if g.get("kick"):
            cfg["env_kwargs"] = {"initial_random": int(g["kick"])}
        if g.get("victim"):
            cfg["victim_algo"] = g["victim"]
        cfg.update(seed=int(g["seed"]), experiment=experiment, config_name=config_name, legacy_name=old_name)
        if episodes:
            cfg["max_episodes"] = episodes
        return experiment, cfg
    return None, None


def legacy_scores(log_dir):
    """old run name -> {'route_count', 'episodes', 'greedy_accuracy'} parsed from the SLURM logs."""
    out = {}
    for path in glob.glob(os.path.join(log_dir, "fir_*.out")):
        text = open(path, errors="replace").read()
        m = re.search(r"task \d+: (\S+) seed=(\d+)", text)
        route = re.search(r"exact target path at final attack step: (\d+)/(\d+)", text)
        greedy = re.search(r"greedy-acc\s+mean over last 5 steps: ([\d.]+)", text)
        if m and route and "### " not in text.split("exact target path")[0][-2000:].replace("### eval", ""):
            out[f"fir_{m.group(1)}_s{m.group(2)}"] = dict(
                route_count=int(route.group(1)), episodes=int(route.group(2)),
                greedy_accuracy_last5=float(greedy.group(1)) if greedy else None)
        # evaluation jobs print one "### <name>" header per run
        for name, body in re.findall(r"^### (?:eval )?(\S+)\n(.*?)(?=^### |\Z)", text, re.S | re.M):
            route = re.search(r"exact target path at final attack step: (\d+)/(\d+)", body)
            greedy = re.search(r"greedy-acc\s+mean over last 5 steps: ([\d.]+)", body)
            if route:
                key = name if name.startswith("fir_") else f"fir_{name}"
                key = re.sub(r"^fir_(v6_cem)_s", r"fir_\1_s", key)
                out.setdefault(key, dict(route_count=int(route.group(1)), episodes=int(route.group(2)),
                                         greedy_accuracy_last5=float(greedy.group(1)) if greedy else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from naming import run_name
    from publish import publish

    scores = legacy_scores(a.logs)
    named = skipped = 0
    for old in sorted(os.listdir(a.results)):
        run_dir = os.path.join(a.results, old)
        experiment, cfg = describe(old)
        if cfg is None or not os.path.isdir(run_dir):
            skipped += 1
            continue
        cfg_path = os.path.join(run_dir, "run_config.json")
        saved = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        merged = {**saved, **{k: v for k, v in cfg.items() if k not in saved or k in ("experiment", "config_name", "env_kwargs", "legacy_name")}}
        merged.setdefault("code_commit", "pre-cleanup (tag pre-cleanup-2026-11-16)")
        merged["run_name"] = run_name(merged)
        if old in scores and not os.path.exists(os.path.join(run_dir, "score.json")):
            merged["legacy_eval"] = scores[old]
        print(f"{old:36s} -> {experiment}/{merged['run_name']}"
              f"{'  [legacy eval ' + str(scores[old]['route_count']) + '/20]' if old in scores else ''}")
        named += 1
        if a.dry_run:
            continue
        json.dump(merged, open(cfg_path, "w"), indent=1)
        if os.path.exists(os.path.join(run_dir, "PUBLISHED")):
            continue
        publish(run_dir)
    print(f"{named} runs named, {skipped} folders skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
