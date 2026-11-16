# State of the project

Last updated: 2026-11-16. Internal working notes: everything a newcomer needs that is
not derivable from the code. Update this file when the state changes.

## 1. One-paragraph summary

The black-box attack works reliably on the **ring** (any size, Q-learning and SARSA
victims) and on the **4x4 grid** when the best of a few seeds is kept. It works on
some seeds on **5x5 and 6x6**, and not yet on **8x8**. On **LunarLander** the pipeline
runs end to end but the experiment as budgeted does not test the attack, because the
victims never learn to fly. **REINFORCE victims** are too slow to finish. The code is
cleaned for release; publishing to Hugging Face is implemented but blocked on a token
with write access to the `envpois` organisation.

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
victim unless stated. Raw lines for every run: `notes/results/raw_fir_log_lines.txt`.

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

Not finished when last checked: grid 5x5 / 6x6 with Q-learning and SARSA (40-95%
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
5. **5x5**: no setting is clearly best (2/4 at most); more seeds needed.
6. **Publish** the existing runs once the Hugging Face token works (section 8).
7. **LLM attacker** idea: `notes/llm_attacker_idea.md` (not built).

## 7. Cluster

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

## 8. Hugging Face (blocked on access)

- Target: organisation `envpois`; repos `attackers`, `behaviour-encoders` (models),
  `results` (dataset). Private until the paper is public.
- Status 2026-11-16: the user `kunwar45` is a member, but the stored OAuth token gets
  `403 You don't have the rights to create a model under the namespace "envpois"`.
  Nothing has been uploaded. fir has no Hugging Face login at all.
- To unblock: a token with write access to `envpois` on the laptop (`hf auth login`,
  granting the organisation) and on fir (`hf auth login` there, or `HF_TOKEN` in the
  job environment).
- Then:
  ```bash
  hf repo create envpois/attackers --repo-type model --private
  hf repo create envpois/behaviour-encoders --repo-type model --private
  hf repo create envpois/results --repo-type dataset --private
  # existing runs, renamed to the convention (run on fir; --dry-run first):
  PYTHONPATH=~/scratch/thesis_sep/release/BaseAttacker python \
    ~/scratch/thesis_sep/release/notes/legacy/publish_legacy_runs.py
  # autoencoders and static data: upload BaseAttacker/ae/models/* to behaviour-encoders,
  # notes/results/*, notes/writeup_*.html and envs/targets/* to results
  ```
- New cluster runs publish themselves (`scripts/slurm/common.sh` -> `publish.py`).

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
| All trained runs and logs | fir scratch (section 7) | to be published to Hugging Face |
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
