"""Publish finished runs to the Hugging Face Hub.

Two repositories, both addressed by the run naming convention (naming.py):

  attackers   (model repo)    <experiment>/<run name>/   best checkpoint, encoder
                              weights, run_config.json, score.json
  results     (dataset repo)  runs/<experiment>/<run name>.json   one record per run:
                              settings, scores, code version, where the checkpoint is
                              logs/<experiment>/<run name>.log    training log, if given

    python publish.py --run_dir results/sweep-3x3x3/grid3d-5x5__sarsa__default__seed0
    python publish.py --pending results          # everything not yet published

Environment variables:
  HF_ATTACKERS_REPO   default envpois/attackers
  HF_RESULTS_REPO     default envpois/results
  HF_PUBLISH=0        switch publishing off
  HF_TOKEN            access token (or log in once with `hf auth login`)

Publishing never fails a run: if the Hub cannot be reached or the token is missing,
the run is marked PUBLISH_PENDING and `--pending` picks it up later.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

ATTACKERS_REPO = os.environ.get("HF_ATTACKERS_REPO", "envpois/attackers")
RESULTS_REPO = os.environ.get("HF_RESULTS_REPO", "envpois/results")
CHECKPOINT_SUFFIXES = ("_actor", "_critic", "_critic2", "_encoder.pt")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return "unknown"


def _checkpoint_files(run_dir: str):
    """The checkpoint that scoring used: best_model_* if present, else the latest numbered one."""
    files = os.listdir(run_dir)
    stem = "best_model"
    if "best_model_actor" not in files:
        numbered = sorted(int(f.split("_")[0]) for f in files
                          if f.endswith("_actor") and f.split("_")[0].isdigit())
        if not numbered:
            return []
        stem = str(numbered[-1])
    keep = [f for f in files if f.startswith(stem + "_") and f.endswith(CHECKPOINT_SUFFIXES)]
    if stem != "best_model" and "latest_encoder.pt" in files:
        keep.append("latest_encoder.pt")
    return sorted(keep)


def record_for(run_dir: str) -> dict:
    """Everything known about a run, as one flat-ish JSON record."""
    with open(os.path.join(run_dir, "run_config.json")) as fh:
        cfg = json.load(fh)
    score_path = os.path.join(run_dir, "score.json")
    score = json.load(open(score_path)) if os.path.exists(score_path) else None
    if score:
        score.pop("per_victim", None)
        score.pop("run_dir", None)
    return {
        "experiment": cfg.get("experiment", "adhoc"),
        "run_name": cfg.get("run_name") or os.path.basename(os.path.normpath(run_dir)),
        "config": cfg,
        "score": score,
        "checkpoint_files": _checkpoint_files(run_dir),
        "attackers_repo": ATTACKERS_REPO,
        "code_commit": cfg.get("code_commit") or _git_commit(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def publish(run_dir: str, log_path: str = None) -> bool:
    """Upload one run. Returns True on success; never raises."""
    run_dir = os.path.normpath(run_dir)
    if os.environ.get("HF_PUBLISH", "1") in ("0", "false", "no"):
        print(f"publish: switched off (HF_PUBLISH=0), skipping {run_dir}")
        return False
    pending = os.path.join(run_dir, "PUBLISH_PENDING")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi()
        record = record_for(run_dir)
        hub_path = f"{record['experiment']}/{record['run_name']}"

        files = record["checkpoint_files"] + [f for f in ("run_config.json", "score.json")
                                              if os.path.exists(os.path.join(run_dir, f))]
        api.create_commit(
            repo_id=ATTACKERS_REPO, repo_type="model",
            operations=[CommitOperationAdd(f"{hub_path}/{f}", os.path.join(run_dir, f)) for f in files],
            commit_message=f"Add {hub_path}")

        operations = [CommitOperationAdd(f"runs/{hub_path}.json",
                                         json.dumps(record, indent=1).encode())]
        if log_path and os.path.exists(log_path):
            operations.append(CommitOperationAdd(f"logs/{hub_path}.log", log_path))
        api.create_commit(repo_id=RESULTS_REPO, repo_type="dataset", operations=operations,
                          commit_message=f"Add {hub_path}")

        with open(os.path.join(run_dir, "PUBLISHED"), "w") as fh:
            json.dump({"attackers": f"https://huggingface.co/{ATTACKERS_REPO}/tree/main/{hub_path}",
                       "results": f"https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/runs/{hub_path}.json",
                       "published_at": record["published_at"]}, fh, indent=1)
        if os.path.exists(pending):
            os.remove(pending)
        print(f"publish: {hub_path} -> {ATTACKERS_REPO} and {RESULTS_REPO}")
        return True
    except Exception as exc:                      # offline node, missing token, ...
        with open(pending, "w") as fh:
            fh.write(f"{type(exc).__name__}: {exc}\n{log_path or ''}\n")
        print(f"publish: could not upload {run_dir} ({type(exc).__name__}: {str(exc)[:200]}). "
              f"Marked PUBLISH_PENDING; retry with: python publish.py --pending <results folder>")
        return False


def publish_pending(root: str) -> None:
    """Publish every run under `root` that has a run_config.json and no PUBLISHED marker."""
    done = failed = 0
    for folder, _dirs, files in sorted(os.walk(root)):
        if "run_config.json" in files and "PUBLISHED" not in files:
            log = None
            marker = os.path.join(folder, "PUBLISH_PENDING")
            if os.path.exists(marker):
                lines = open(marker).read().splitlines()
                log = lines[1] if len(lines) > 1 and lines[1] else None
            ok = publish(folder, log)
            done, failed = done + ok, failed + (not ok)
    print(f"publish: {done} uploaded, {failed} still pending")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir")
    ap.add_argument("--log", default=None, help="training log to upload alongside")
    ap.add_argument("--pending", metavar="RESULTS_DIR", default=None)
    a = ap.parse_args()
    if a.pending:
        publish_pending(a.pending)
    elif a.run_dir:
        publish(a.run_dir, a.log)
    else:
        ap.error("give --run_dir or --pending")
    sys.exit(0)                                   # publishing never fails the job


if __name__ == "__main__":
    main()
