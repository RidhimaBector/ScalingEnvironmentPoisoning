# CLAUDE.md

Guidance for Claude Code (and any new collaborator) working in this repository.
Read this first, then `notes/STATE.md` for where the research stands.

## What this is

Black-box **environment-poisoning attacks** on reinforcement learners: an attacker
changes a victim's environment (never the victim) and learns, from the victim's
observed behaviour only, how to steer it into a target behaviour. Extension of
"Spiking Pitch Black" (AAMAS 2022) toward a journal paper ("Transferable Environment
Poisoning on Reinforcement Learning: Attack and Resilience").

The code must stay **environment-agnostic and victim-agnostic**: the attack pipeline
talks to games and learners only through the interfaces in `BaseAttacker/core/`.
A change that makes the attacker depend on what a perturbation means is a bug.

## Where things live

| Thing | Location |
|---|---|
| Code | `BaseAttacker/` (see `README.md` for the layout and how to extend it) |
| Current research state, results, open problems | `notes/STATE.md` |
| Raw per-run result lines from the cluster | `notes/run_summaries/raw_fir_log_lines.txt` |
| Paper (separate git repo, Overleaf) | `paper/` (ignored here; see below) |
| Trained attackers, autoencoders, results | Hugging Face org `envpois` (see below) |
| Cluster working copy | `fir:~/scratch/thesis_sep/` |

## Commands

All from `BaseAttacker/`. The Python environment is `../venv` (pip + venv; **never conda**).

```bash
../venv/bin/python -m pytest tests -q                       # 10 s; run before every commit
../venv/bin/python main.py with quick_test                  # smoke run, seconds
../venv/bin/python main.py with seed=0 experiment=demo      # real run, ~35 min
../venv/bin/python score_attack.py --run_dir results/<experiment>/<run>
../venv/bin/python plot_grid_attack.py --run_dir results/<experiment>/<run>
../venv/bin/python publish.py --pending results             # upload unpublished runs
```

Settings and defaults: `BaseAttacker/defaults.py`. Run names: `BaseAttacker/naming.py`
(`<experiment>/<env>-<variant>__<victim>__<config>__seed<N>`); use it for every folder
and Hub path, never an ad hoc name.

## Rules of engagement

- **Training runs go on the cluster, not the laptop** (the laptop overheats). Short
  smoke tests and scoring a few victims locally are fine.
- **Code reaches the cluster by git** (`git push`, then `git pull` there). A hook blocks
  rsync and in-place edits of tracked files over SSH; it also misfires when one shell
  command both writes a file with a heredoc and calls `ssh`, so do those separately.
- Foreground `sleep N; cmd` is blocked: wait with a background until-loop instead.
- A hook asks you to look at every generated plot: do it.
- **Report results honestly**: every seed, the no-attack baseline, and what was not
  verified. Do not call a seed-selected result a single-run result.
- Before a full sweep, smoke-test the exact command on a few episodes.
- After changing code, list the choices the user did not specify.
- LunarLander needs Box2D, which does not build on the laptop's Python 3.14: it only
  runs on the cluster. Its tests are skipped locally.
- Sacred hangs after a crash if a run's heartbeat thread is left alive; `main.py`
  closes the victim run in a `try/except BaseException` for this reason. Keep it.

## Cluster (fir, Alliance Canada)

- `ssh fir` (needs password + Duo; the user must log in, then a control socket is
  reused for about 4 hours). If SSH fails with "Permission denied", ask the user to
  run `ssh fir` in a terminal.
- SLURM account `def-s2ganapa`. Compute nodes have internet (Hugging Face reachable).
- `~/scratch/thesis_sep/ScalingEnvironmentPoisoning` is a checkout of the
  **pre-cleanup** code (tag `pre-cleanup-2026-11-16`); the November 2026 results in its
  `BaseAttacker/results/` and `logs/` were produced by it. Do not `git pull` there:
  clone the current code next to it instead (see `notes/STATE.md`, "Cluster").
- venv: `~/scratch/thesis_sep/venv` (`module load python/3.12` first). It needs
  `setuptools<81` for Sacred. Box2D comes from the Alliance wheelhouse and exposes its
  classes under `Box2D.Box2D` (handled in `envs/lunar_lander.py`).
- Sweeps: `scripts/slurm/*.sh`, always with `--account=def-s2ganapa` and
  `PYTHON_MODULE=python/3.12 VENV=~/scratch/thesis_sep/venv`.
- Jobs longer than the time left before a maintenance reservation stay pending with
  reason `ReqNodeNotAvail, Reserved for maintenance`; shorten `--time`.

## Cluster (killarney, Alliance Canada)

- Second cluster; the sweep scripts are cluster-neutral (`BaseAttacker/scripts/slurm/README.md`).
- Account `aip-s2ganapa`. Code and venv under `/project/aip-s2ganapa/kunwar/`; outputs
  under `$SCRATCH` via `RESULTS_ROOT=$SCRATCH/envpois/results`.
- Each SSH session needs a Duo approval on the user's phone (tell them before
  connecting) and lasts about an hour. SLURM commands and `module` need a login shell:
  wrap them in `bash -lc "..."`.
- First-time setup there: `bash scripts/setup_alliance.sh <venv path>` from `BaseAttacker/`.

## Hugging Face

Organisation `envpois`, three repositories (private until the paper is out):

| Repo | Type | Layout |
|---|---|---|
| `envpois/attackers` | model | `<experiment>/<run name>/` checkpoint, `run_config.json`, `score.json` |
| `envpois/behaviour-encoders` | model | pre-trained trace autoencoders |
| `envpois/results` | dataset | `runs/<experiment>/<run name>.json`, `logs/...` |

Cluster runs publish themselves at the end of `train_and_score` (`publish.py`); a
failed upload marks the run `PUBLISH_PENDING` and never fails the job. Publishing needs
a token with write access to `envpois` (`hf auth login` or `HF_TOKEN`).
The laptop has such a token; fir does not (as of 2026-11-16). Never read, print or copy
a token yourself: logging in is the user's step.

## Paper

`paper/` is a clone of the Overleaf project (its own git remote; ignored by this
repo). `prelim_method.tex` is a stand-alone methodology document with its diagrams in
`paper/adhoc/`. Commit there and `git push` to update Overleaf; do not edit `main.tex`
without being asked (it belongs to the co-authors).

## Working with the user

- Lead with the result or the next action; number steps; give concrete time estimates;
  restate where things stand each turn; no preamble or closing pleasantries.
- Ask before anything hard to undo or outward-facing that was not requested.
- Keep `main` current: commit and push finished work.
