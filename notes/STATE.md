# State of the project

Last updated: 2026-11-16. Internal working notes: everything a newcomer needs that is
not derivable from the code. Update this file when the state changes.

## 1. One-paragraph summary

The black-box attack works reliably on the **ring** (any size, Q-learning and SARSA
victims) and on the **4x4 grid** when the best of a few seeds is kept. It works on
some seeds on **5x5 and 6x6**, and not yet on **8x8**. On **LunarLander** the pipeline
runs end to end but the experiment as budgeted does not test the attack, because the
victims never learn to fly. **REINFORCE victims** are too slow to finish. The code is
cleaned for release, and all existing runs, results and autoencoders are published
(privately) to the `envpois` organisation on Hugging Face.

## 2. Definitions used everywhere

- **Attack episode**: fresh victim, natural environment, `I` attack epochs (15; 10 on
  the lander). **Attack epoch**: attacker perturbs, victim relearns `K` episodes (80;
  20 on the lander), attacker is rewarded.
- **Exact route / `path_match`**: the victim's greedy route from the start equals the
  target route. **Target match / `greedy_accuracy`**: share of target states whose
  greedy action is the target's. Both are ground truth, used only for evaluation.
- **Poisoned victims of 20**: a trained attacker is run without exploration against 20
  fresh victims; count of victims on the exact route at the last epoch.
- **A seed "works"**: at least 11 of 20 victims poisoned. (This threshold is ours.)
- **`best_avg_reward`**: best 50-episode moving average of the attacker's training
  return. It is black-box-observable and is used to pick checkpoints and seeds.
- **Push episode**: every perturbation dimension driven one way for the whole episode.
  **Adaptive push**: directions sampled from probabilities updated by the
  cross-entropy method (`attack/push_exploration.py`).
- **`obs_dynamics`**: `env` = attacker also sees the environment's dynamics vector
  (rounds 1-9); `actions` = it sees only the running sum of its own actions (round 10
  onward, the default now).

## 3. What was done, in order

All runs on fir, 1 CPU each. "k/20" = poisoned victims of 20. 4x4 grid and Q-learning
victim unless stated. Raw lines for every run: `notes/run_summaries/raw_fir_log_lines.txt`.

| Round | Change tested | Result |
|---|---|---|
| 0 | Got the pipeline running: Sacred, AE encoder, reward fixes | First working local attacker (old AE, trace reward): 14/20 |
| 1 | Encoder and reward (3000 ep, 3 seeds) | Original AE 0/3 seeds; **retrained AE + trace reward 2/3** (17, 14); trajectory reward 0/3 (target match ~0.85 but never exact); LSTM 0/3 |
| 2 | Bundle: twin critics + normalised input + push + long warm-up (5000 ep) | Worse: 1/3 for two configs, rest 0/3 |
| 3 | One change at a time (3000 ep, 3 seeds) | base (new seeds) 0/3; twin critics 0/3; normalised input 1/3; noise 0.2 0/3; mix reward 1/3 (19) + one 7; **uniform push 1/3 (20/20)** |
| 4 | Uniform push, 12 more seeds | 3/12 (15, 20, 13) |
| 5 | Adaptive push and elite imitation (3000 ep, 6 seeds) | adaptive 1/6; imitation alone 0/6; both 2/6. Push search often found the answer while the policy had not absorbed it |
| 6 | Same, 6000 ep | **adaptive push 5/6** (20, 16, 20, 0, 18, 17); with imitation 3/6 |
| 7 | Adaptive push, 12 more seeds | 4/12 (+ one at 10). **Combined 9/18** |
| 8 | 5x5 and 6x6, 4x4 settings, 8000 ep | 5x5: 1/6 (20) + one 9; 6x6: 0/6. Target match 0.80-0.96 |
| 9 | Grid-size tuning: push floor scaled to grid; push 20% / 40% / 40% + batch 32 (12000 ep, 4 seeds each) | 5x5: 20%: 1 (20) + 9, 7; 40%: 1 (20); batch32: 2 (19, 17). 6x6: 20%: 0/4; 40%: 2/4 (20, 19); batch32: 2/4 (18, 20). **8x8: 0/12** (target match 0.55-0.92) |
| 10 | `obs_dynamics=actions` (agnostic observation) | 4x4 Q-learning 5/12 (+ one at 10); **4x4 SARSA 3/4**; **ring-8 Q-learning 4/4 and SARSA 4/4, all 20/20** |
| 11 | 3 x 3 x 3 sweep (3 seeds per cell), see section 4 | in progress |

Engineering done along the way: retrained trace autoencoder (72% reconstruction vs
47%); victim/env sampling 1.8x faster; Sacred logging thinned (36 MB -> 7 MB per run);
arbitrary rows x cols grids; `ring` and `lunarlander` environments; DQN, VPG, A2C
victims; uniform scoring with a no-attack baseline; release clean-up; naming and
publishing.

## 4. The 3 x 3 x 3 sweep (round 11), status at `2026-11-16 14:05 MST` (cluster time)

SLURM jobs on fir: `4423670` (grid and ring cells, tasks 0-53), `4429537` (reruns of
tasks 50, 51 after a Lustre I/O error), `4423671` (lander cells, tasks 54-80,
training capped at 13 h). Old-style run folders: `results/fir_s27_<cell>_s<seed>`.

Finished (poisoned of 20, per seed; baseline was 0/20 in every case):

| Cell | Seeds |
|---|---|
| grid 4x4, Q-learning | 16, 11, 19 |
| grid 4x4, SARSA | 0, 11, 16 |
| ring 8, Q-learning / SARSA | 20, 20, 20 / 20, 20, 20 |
| ring 12, Q-learning / SARSA | 20, 20, 20 / 20, 19, 17 |
| ring 16, Q-learning / SARSA | 20, 20, 10 / 20, 20, 20 |
| grid 5x5, Q-learning | 0 (target match 0.88), two seeds still running |
| lander, VPG and A2C (18 runs) | no effect: target match and similarity unchanged within noise; victim return -108 to -200 |

Added at `2026-11-16 17:00 MST` (cluster time), after the Hugging Face snapshot:

| Cell | Seeds (poisoned of 20; target match without -> with attack) |
|---|---|
| grid 5x5, Q-learning | 0, 0, 0 (0.48 -> 0.88, 0.83, 0.84) |
| grid 5x5, SARSA | 0, 0, 0 (0.45 -> 0.92, 0.76, 0.64) |
| grid 6x6, Q-learning | 0, 0, 0 (0.50 -> 0.64, 0.61, 0.54) |
| grid 6x6, SARSA | 0, 0, 0 (0.47 -> 0.65, 0.71, 0.73) |
| ring 8, REINFORCE | 6, 7, 5, against a baseline of 6, 6, 6: no effect |

**The larger grids regressed.** With the tuned push setting but the own-actions
observation (`obs_dynamics=actions`), 0 of 12 runs reached the exact route on 5x5 and
6x6; the same push setting with `obs_dynamics=env` gave 2/4 and 2/4 in round 9. Either
the attacker needs the absolute dynamics on larger grids, or this is seeds; it has not
been isolated. Test: rerun the round-9 batch-32 setting on 6x6 with both observations
and the same seeds.

Still running then: the 9 grid REINFORCE cells (about 800-1300 episodes done; they hit
the 8 h limit unscored), ring-16 REINFORCE seed 0 (finishing), and the 9 lander DQN
cells (110-260 of 1000 episodes; training stops at 13 h, then they are scored).

### Final state of the sweep, `2026-11-17 04:20 MST` (cluster time)

All three jobs have ended; nothing of this project is queued on fir. 68 of 81 cells
have a score; 13 timed out unscored.

| Cell | Seeds (poisoned of 20, baseline in brackets; target match without -> with attack) |
|---|---|
| ring 12, REINFORCE | 0, 1, 1 (baseline 2, 2, 2); 0.47 -> 0.45, 0.47, 0.52: no effect |
| ring 16, REINFORCE | 0, 0, 0 (baseline 0); 0.39 -> 0.50, 0.44, 0.44. Seeds 1 and 2 were scored from checkpoints at 1019 and 2059 of 3000 episodes |
| ring 16, SARSA seed 2 (rerun) | 20 |
| lander 500, DQN | seeds 1, 2 scored at 512 and 487 of 1000 episodes: target match 0.39 -> 0.46, 0.33 -> 0.32; similarity 0.84 -> 0.68, 0.87 -> 0.63; victim return -103 -> -119, -94 -> -141. Seed 0 timed out |
| lander 1000, DQN | seeds 0, 1 at 556 and 523 episodes: target match 0.55 -> 0.41, 0.51 -> 0.57; similarity 0.82 -> 0.67, 0.83 -> 0.71; victim return -106 -> -108, -86 -> -106. Seed 2 timed out |
| lander 1500, DQN | seed 1 at 449 episodes: target match 0.47 -> 0.54; similarity 0.83 -> 0.72; victim return -92 -> -123. Seeds 0, 2 timed out |

Read of the lander DQN cells: similarity to the target *falls* under attack in all 5
scored runs (by 0.11 to 0.24) and the victim's return falls in all 5; target match
moves both ways. The attack changes these victims but not towards the target. The
victims are still untrained (returns near -100), so this is not a valid test either.

Timed out unscored (13): the 9 grid REINFORCE cells (tasks 6-8, 15-17, 24-26 of
`4423670`, 8 h limit) and 4 lander DQN cells (tasks 54, 65, 72, 74 of `4423671`, 15 h
limit, reached during scoring). Their checkpoints are in the old checkout on fir and
can be scored by hand with the command below.

Totals by victim family, counting a seed as working at 11 or more of 20:
grid 4x4 Q-learning/SARSA 5 of 6; grid 5x5 and 6x6 0 of 12; ring Q-learning/SARSA
17 of 18 (the miss is 10/20); ring REINFORCE 0 of 9; lander 0 of 23 scored.

### Round 12: observation comparison on killarney, submitted `2026-11-17 06:26 EST` (cluster time)

Job `7710630`, tasks 0-15, `scripts/slurm/sweep_obs_comparison.sh` at commit `200111a`:
grid 5x5 and 6x6, Q-learning, push setting 0.4 / batch 32 / elite 8 / rate 0.3, 12,000
episodes, seeds 0-3, each with `obs_dynamics=env` and `obs_dynamics=actions`.
`TRAIN_TIMEOUT=14h`, job limit 16 h. Results:
`$SCRATCH/envpois/results/obs-comparison/grid3d-<N>x<N>__qlearning__obs-<env|actions>__seed<k>`;
logs `BaseAttacker/logs/obscompare_7710630_<task>.out` in the killarney checkout. Runs
will end `PUBLISH_PENDING` (no Hugging Face token there). A 60-episode smoke run (job
`7710471`, tasks 3 and 12) trained and scored in about 2 minutes; its output was deleted.
Collect with: `grep -h "^run:\|^SCORE" logs/obscompare_7710630_*.out`.

Not finished when last checked (earlier note): grid 5x5 / 6x6 with Q-learning and SARSA (40-95%
done); every REINFORCE cell (grid: ~300 of 6000-12000 episodes; ring: 20-95%); lander
DQN (~50 of 1000 episodes). The grid/ring job has an 8 h limit and the default 21 h
training timeout, so **unfinished REINFORCE cells are killed before scoring**: score
their saved checkpoints by hand afterwards.

**To collect what remains** (on fir, old checkout):

```bash
cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker
grep -h "^task\|^SCORE" logs/fir_4423670_*.out logs/fir_4429537_*.out logs/fir_4423671_*.out
# cells without a SCORE line: python score_attack.py --run_dir results/fir_s27_<cell>_s<seed> --episodes 20
```

## 5. What we believe, and how sure

1. **Seed variance dominates.** Outcomes are bimodal: a seed fails outright or poisons
   most victims. No learner change tried fixes this. (Strong: ~150 runs.)
2. **Training reward predicts success on 4x4.** Without imitation, all 18 runs with
   `best_avg_reward >= 12.5` poisoned >= 10/20 and all 36 below poisoned none (rounds
   1-7). So "train several seeds, keep the highest-reward one" is a valid black-box
   procedure. On 5x5 the threshold is less clean (one run at 13.37 scored 0/20), and
   with imitation it fails. Rewards are not comparable across grid sizes.
3. **Training length matters.** Adaptive push went from 1/6 (3000 ep) to 5/6 (6000 ep)
   on the first six seeds; the next twelve gave 4/12, so "5/6" was partly luck.
4. **Bigger grids need more search.** With a fixed probability floor each cell strays
   from its best direction 10% of the time; on 36 cells only ~2% of pushes try the best
   map. Scaling the floor (`cem_floor=-1`) and pushing in 40% of episodes got 6x6 from
   0/6 to 4/8.
5. **The target is reachable on every size** with a hand-set map
   (`scripts/probe_grid_maps.py`): 4x4 1.00, 5x5 1.00, 6x6 0.65, 8x8 0.68, 10x10 0.85.
   The map the 4x4 attacker learns (perimeter 10, interior 0) scores 0.00 on 8x8
   because a flat low interior is a shortcut; an interior with barriers closes it. So
   on 8x8 the attacker, not the environment, is the limit. (8 victims per map: rough.)
6. **Elite imitation, twin critics, input normalisation and more noise do not help.**
7. **The lander test is invalid as budgeted.** 200 victim episodes per attack episode
   is too few; the feasibility probe (`scripts/probe_lander.py`, job 4434606) shows
   un-attacked victims reach returns of +130 to +260 only after 600-3000 episodes.
   The same probe shows hand-set poisoning does change a trained victim: failing all
   engines drops the return to about -115 and main-engine use to ~0; failing the left
   half moves the final horizontal position right (DQN: +0.26, +0.42; the target ends
   at +0.34). So the lander is attackable in principle, but each attack epoch needs
   hundreds of victim episodes, or victims that are pre-trained and then adapted.
   Also: embedding similarity is 0.45-0.80 with no attack, and "target match" is 0.57
   for a victim that always fires the main engine, so neither is a usable success
   measure alone. Probe was still running when recorded (some seeds missing).
8. **REINFORCE victims** take ~25x longer per attack episode on the grid than
   Q-learning (long wandering episodes). They need a step cap or a much longer budget.

## 6. Open problems, most valuable first

1. **Lander**: redesign the budget (more victim episodes per epoch, or warm-started
   victims), choose a meaningful success measure (e.g. final horizontal position vs
   the target's), then rerun. Finish reading the probe first.
2. **Reliability**: decide and state the claim. A single run works about half the time
   on 4x4 and 6x6; "best of N seeds by training reward" is what is dependable. The
   user's target is "80%+ on all sizes" (unclear whether of victims or of seeds; ask).
3. **8x8**: 0/12. Options not tried: much longer training, push-only search with
   averaging over repeats, curricula from smaller grids, a reward that sees whole-route
   completion.
4. **REINFORCE cells**: cap victim episode length or drop them from the grid.
5. **5x5 and 6x6 with the own-actions observation: 0/12** in the sweep (section 4).
   Isolate observation versus seeds before anything else on larger grids.
6. **Refresh the Hugging Face snapshot** when the remaining sweep cells are scored (section 8).
7. **LLM attacker** idea: `notes/llm_attacker_idea.md` (not built).

## 7. Clusters

fir is where everything so far ran. killarney is the second cluster; see the end of this section.

- Pre-cleanup checkout (do not pull): `~/scratch/thesis_sep/ScalingEnvironmentPoisoning`,
  at commit `d39baf7`-era code (tag `pre-cleanup-2026-11-16`). Holds `BaseAttacker/results/`
  (5.2 GB, 255 run folders), `logs/`, `sacred_runs/`, and trained autoencoders.
- A second, detached checkout used once to test the release branch:
  `~/scratch/thesis_sep/release_check` (safe to delete).
- For new work, clone the current `main` to `~/scratch/thesis_sep/release` and run
  sweeps from its `BaseAttacker/`:
  ```bash
  cd ~/scratch/thesis_sep && git clone <remote> release && cd release/BaseAttacker && mkdir -p logs
  PYTHON_MODULE=python/3.12 VENV=~/scratch/thesis_sep/venv \
    sbatch --account=def-s2ganapa --array=0-80 scripts/slurm/sweep_3x3x3.sh
  ```
  The venv there needs `pip install --no-index huggingface_hub pytest`.
- **Outage**: reservation `NovemberOutage` starts `2026-11-17 07:00 MST` (cluster time) with no published
  end. Jobs still running then are killed. Scratch is not backed up and may be purged.
- Typical speed: 4x4, 6000 episodes: ~35 min. 6x6, 12000: ~5 h. 8x8, 12000: ~8 h.
  Ring, 3000: ~10 min. Lander VPG/A2C, 1000 episodes at K=20: ~45 min; DQN ~25 h.

**killarney** (Alliance; set up and verified 2026-11-16):

- Account `aip-s2ganapa`. Code: `/project/aip-s2ganapa/kunwar/envpois/ScalingEnvironmentPoisoning`.
  venv: `/project/aip-s2ganapa/kunwar/venvs/envpois` (built with `scripts/setup_alliance.sh`;
  torch 2.14, Box2D present). Results: `$SCRATCH/envpois/results` (= `/scratch/kunwar/...`,
  NFS, purge-eligible).
- Verified there: 27 tests pass (lander included); a 2-task smoke run of
  `sweep_ablation_4x4.sh` trained, scored and attempted to publish (job 7679356,
  about 2.5 min per task for 120 episodes). Its output folders were deleted afterwards.
- Submit from `BaseAttacker/` in a login shell:
  ```bash
  export PYTHON_MODULE=python/3.12 VENV=/project/aip-s2ganapa/kunwar/venvs/envpois RESULTS_ROOT=$SCRATCH/envpois/results
  sbatch --account=aip-s2ganapa --array=<range> scripts/slurm/<sweep>.sh
  ```
- POT does not build there, so it is not installed; only `privacy_mode=full_whitebox`
  needs it (the code imports it lazily and the white-box test is skipped).
- No Hugging Face token there either: runs end as `PUBLISH_PENDING`.
- SSH needs a Duo tap from the user and the connection lasts about an hour. `sbatch`,
  `squeue`, `sacct` and `module` only exist in a login shell (`bash -lc "..."`).
  `git pull` over HTTPS occasionally fails with "Broken pipe"; retry.

## 8. Hugging Face

Organisation `envpois`, three private repositories, uploaded 2026-11-16 from the laptop:

| Repository | Content |
|---|---|
| `envpois/attackers` (model) | 255 run folders in 10 experiments; 249 have a checkpoint (actor + critic), all have `run_config.json`, 49 have `score.json` |
| `envpois/results` (dataset) | `runs.jsonl` (255 rows, 223 with an evaluation), one JSON record and one log per run, the raw summary file, the lander target, a copy of this file and the write-up |
| `envpois/behaviour-encoders` (model) | the 8 trace autoencoders under `<env>-<variant>__trace-autoencoder.pt` |

How the snapshot was built (repeat it to refresh after more runs finish on fir):
rsync `results/*/{best_model_actor,best_model_critic,best_model_critic2,best_model_encoder.pt,run_config.json,score.json}`
and `logs/fir_*.out` from the old fir checkout to a local folder, run
`notes/legacy/publish_legacy_runs.py --stage <dir>` from that folder with the current
`BaseAttacker` on `PYTHONPATH`, then `hf upload` the staged `attackers/` and `results/`
folders. The 12 round-1 runs took their evaluation from eval job 4254633, whose log has
no per-run headers; they were matched by position (seed 0-2, each: original-ae-trace,
retrained-ae-trace, retrained-ae-traj, lstm-trace).

Not in the snapshot: sweep-3x3x3 cells that had not been scored yet (grid 5x5/6x6,
REINFORCE, lander DQN), and the feasibility-probe output beyond what is in the summary
file. The repository cards (README) were written by hand at upload time.

**fir has no Hugging Face token** (the user logged in on the laptop only), so jobs
there mark their runs `PUBLISH_PENDING`. Either log in on the cluster once
(`hf auth login` inside the venv) or publish from the laptop as above. New-layout runs
are picked up by `python publish.py --pending <results folder>`.

## 9. Paper and write-ups

- Overleaf project cloned at `paper/` (remote `git.overleaf.com/6a0e0b6591234bf44281cbd1`).
  `prelim_method.tex` (stand-alone methodology, 15 references) and four TikZ diagrams
  in `paper/adhoc/` were pushed on 2026-11-16. Nine bibliography entries were written
  from memory into `ref_kunwar.bib`: verify them before submission. `main.tex` is the
  co-authors' and was left untouched.
- `notes/writeup_2026-11-16.html`: a short scientific write-up of rounds 1-9 (source of
  a private claude.ai artifact). It predates rounds 10-11; its 8x8 row says "running".
- The methodology document describes the lander at K=20, I=10, which section 5.7 says
  is not a valid test; update it when the lander budget changes.

## 10. Things that exist only outside git

| What | Where | Note |
|---|---|---|
| Full run folders (optimizer states, numbered checkpoints, Sacred records) | fir scratch (section 7) | the scored checkpoint, settings, score and log of each run are on Hugging Face |
| Old local checkpoints | laptop `BaseAttacker/results/` (329 MB, ignored) | pre-November runs; `nov15_ae_trace_s0` is the first working attacker |
| Pre-2026-11 design notes | laptop `docs/` (ignored by a long-standing rule) | `WALKTHROUGH.org`, `HIERARCHICAL_SACRED_ARCHITECTURE.org`, MongoDB setup |
| Local Sacred MongoDB | `localhost:27017`, db `env_poisoning` | contains a few mis-launched test runs from 2026-11-15 |

## 11. History of the repository

- `main` before 2026-11-16 held the working code used for rounds 0-11; its last commit
  is tagged `pre-cleanup-2026-11-16`. Older lines of work (`kn/run`, `kunwar`,
  `kunwar.environment-encapsulation`, `kunwar.test`, `logit`) were merged into it as
  history only.
- The release clean-up removed 400 MB of tracked CSVs, old figures and legacy scripts,
  split settings / builders / Sacred / training loop, renamed modules
  (`env3D_4x4 -> grid3d`, `chain -> ring`, `victim_Q -> victim_qlearning`,
  `utils_attack -> metrics`, `eval_attack.py -> plot_grid_attack.py`), and added tests,
  the README, the licence (MIT) and the SLURM scripts in `scripts/slurm/`.
- Behaviour changes in the clean-up: Sacred logs to files unless `SACRED_MONGO_URL` is
  set; defaults are the recommended attack; seeding order differs slightly, so runs
  are not bit-identical to pre-cleanup runs with the same seed. Verified equivalent:
  grid transition matrices are identical, and a pre-cleanup checkpoint scores the same.
