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

### Round 13: push base and online autoencoder, submitted `2026-11-17 06:48 EST` (cluster time)

Code at commit `c18da78` adds two options, both off by default:

- `push_base=True`: the attacker's action is (the push search's current best direction
  per dimension, where its probability is >= `push_base_conf`=0.5, else 0) + the
  actor's output, clipped. The critic is unchanged; the base is saved with every
  checkpoint (`<name>_base.npy`). Motivated by the diagnosis below.
- `ae_model_path=online`: no pre-trained autoencoder. The run's first
  `ae_online_episodes`=500 attack episodes are exploration only (random or push); the
  trace autoencoder is then fitted on the traces seen (60 epochs), frozen, and the
  replay buffer is cleared. The fit prints its held-out reconstruction accuracy.

Job `7711713` on killarney, tasks 0-35, `scripts/slurm/sweep_push_base.sh`: grid 4x4
(6000 ep, push 0.2), 5x5 and 6x6 (12000 ep, push 0.4 / 32 / 8 / 0.3), Q-learning,
`obs_dynamics=actions`, seeds 0-3, configs `base`, `online-ae`, `base-online-ae`.
Results: `$SCRATCH/envpois/results/push-base/`; logs `logs/pushbase_7711713_<task>.out`.
Comparison without either option: round 12 config `obs-actions` (5x5, 6x6, same seeds)
and the 3x3x3 sweep (4x4). Smoke run `7711585` (90 episodes) passed; with only 30
episodes of traces the online fit reached 0.53 reconstruction (pre-trained 5x5: 0.66).
Collect with: `grep -h "^run:\|^SCORE\|fitted" logs/pushbase_7711713_*.out`.

### Speed-up and rerun of the 6x6 check, `2026-11-17 07:20 EST` (cluster time)

Commit `a404450`: tabular victims train in numba-compiled loops
(`victim/fast_tabular.py`) when the environment reports a transition table
(`tabular_model()`, implemented for grid and ring). Same update rules and sampling
distributions; the Python loops remain the reference (`ENVPOIS_NO_JIT=1`). Checked:
with 300 victims per path, ring-8 Q-learning episode length is 5.80 (Python) vs 5.79
(compiled); tests compare the two paths. Profile before: 88% of a 6x6 run was the
victim loop. Measured on the laptop, seconds per attack episode before -> after:
6x6 Q-learning 0.46 -> 0.13 (3.6x), 4x4 SARSA 0.34 -> 0.10 (3.3x), ring-8 REINFORCE
2.93 -> 0.14 (21x), 4x4 REINFORCE 15.3 -> 0.42 (37x). On killarney 6x6 went from
about 1.3 to about 0.27 s per episode. What is left is the attacker's own 15 gradient
steps per episode (about 5 ms each, spread over many small tensor operations; more
threads or a GPU do not help at this network size). The lander is not sped up.
numba is installed in the killarney venv (0.65.1); not yet on fir.

Round 13 was cut down at the user's request ("don't need a full sweep until this
works"): job `7711713` cancelled. Replacement job `7712563` (`pushbase-fast`), tasks
32, 33, 36, 37, 40, 41, 44, 45 of `sweep_push_base.sh`: 6x6, seeds 0-1, configs
`plain`, `base`, `online-ae`, `base-online-ae`, 12,000 episodes, about an hour each.
Interim read of the cancelled runs at ~1,500 episodes: the online autoencoder's
held-out reconstruction was 0.58 on all four runs (pre-trained 6x6: 0.59); returns had
not separated between configs yet.
Lander: smoke run `7712063` (DQN, K=60, I=10, 4 attack episodes) was still running
after 35 minutes; `scripts/slurm/verify_lander.sh` is not submitted yet.

**Result of job `7712563`** (all 8 completed, about 1 h each; 20 fresh victims per
run; target match and exact-route count are no attack -> attack):

| Config | Seed | best_avg_reward | Target match | Exact route | Online AE accuracy |
|---|---|---|---|---|---|
| plain | 0 | 13.21 | 0.49 -> 0.93 | 0/20 -> 1/20 | |
| plain | 1 | 12.11 | 0.49 -> 0.90 | 0/20 -> 0/20 | |
| base | 0 | 10.30 | 0.49 -> 0.77 | 0/20 -> 0/20 | |
| base | 1 | 12.87 | 0.49 -> 0.96 | 0/20 -> 0/20 | |
| online-ae | 0 | 9.56 | 0.49 -> 0.59 | 0/20 -> 0/20 | 0.577 |
| online-ae | 1 | 11.74 | 0.49 -> 0.89 | 0/20 -> 0/20 | 0.580 |
| base-online-ae | 0 | 10.93 | 0.49 -> 0.69 | 0/20 -> 0/20 | 0.577 |
| base-online-ae | 1 | 12.57 | 0.49 -> 0.93 | 0/20 -> 0/20 | 0.582 |

Reading: neither change passes the criterion (11 of 20 victims on the exact route) on
6x6; no config does, including `plain`. Mean target match: plain 0.92, base 0.87,
online-ae 0.74, both 0.81 (2 seeds each, so differences between configs are not
established). The online autoencoder matches the pre-trained one in reconstruction
(0.58 vs 0.59). Open: target match of 0.90-0.96 with 0-1 of 20 exact routes means the
victims take the target action in most states but still miss at least one state on
the route; the exact-route measure, not the match, is what fails here.
The paper's method document was not updated (condition "if these work" not met).

**Where the 6x6 victims leave the route** (`scripts/diagnose_grid_route.py`, the 8
saved attackers, 20 fresh victims each, 160 victims; target route is 15 moves: top
row, right column, bottom row):

- 159 of 160 victims are off the exact route at the last attack step, and every one
  breaks within the first 3 cells of the route (positions 0-2 of 15), mostly at the
  start cell, where the target says east and the goal is 5 cells straight south.
- Cause of the first break: wrong greedy action in 129 victims; right action but the
  move is so far uphill that staying put is the most likely outcome (the route measure
  follows the most likely transition) in 30 victims.
- Wrong route actions per victim (of 15), mean: plain 1.00 / 1.50, base 3.40 / 0.65,
  online-ae 6.20 / 1.65, both 4.65 / 1.00 (seed 0 / seed 1).
- Victims with all 15 route actions right: base seed 1: 11 of 20; both seed 1: 6;
  plain seed 0: 1; the other five runs: 0. These count as off-route only because of
  the uphill rule above.
- The learned maps keep the bottom row (part of the route) at altitude 0 and differ
  from the hand-set maps (route at 10).

### Round 14: route-prefix reward, `2026-11-17` (commit `2946482`)

Priority set by the user on 2026-11-17: "getting it working complete black box and
general for gridworld is our first step". The lander is parked.

New reward modes (`defaults.py`, `utils/metrics.py: route_prefix_from_trajectories`),
computed only from the victim's observed (state, action, next state) tuples, most
recent half of them: starting from the first state the victim was seen in, walk forward
while its most frequent action is the target action, moving to the state it most often
reached next (a failed move does not count as a step); the reward is the number of
distinct target states passed divided by the number of target states.
`reward_mode=route` is that alone, `route_mix` is its mean with the trace reward.
The best-checkpoint choice follows the reward, so it changes with the mode.

Check against ground truth on the saved 6x6 attackers (20 victims each; observed
route prefix | true prefix of the greedy policy | trace reward): base seed 0:
0.00 | 0.00 | 0.68; base seed 1: 0.34 | 0.57 | 0.88; plain seed 0: 0.05 | 0.05 | 0.90;
plain seed 1: 0.09 | 0.09 | 0.85. The new reward is low exactly where the trace reward
is high and the route is missed; it underestimates on base seed 1.

Test: `scripts/slurm/sweep_route_reward.sh`, tasks 0, 1, 4, 5 (6x6, seeds 0-1,
`route` and `route-mix`), otherwise the `plain` settings of round 13.

**Result of round 14** (job `7714711`, 6x6, own-actions observation, 20 fresh victims;
diagnosis with `scripts/diagnose_grid_route.py`):

| Mode | Seed | Target match | Exact route, last step | Exact route at any of 15 steps | Share on route per attack step (steps 7-15) | All 15 route actions right, last step |
|---|---|---|---|---|---|---|
| route | 0 | 0.49 -> 0.87 | 1/20 | 16/20 | .60 .20 .15 .10 .05 .05 .05 .10 .05 | 1/20 |
| route | 1 | 0.49 -> 1.00 | 4/20 | 20/20 | .00 .00 1.0 1.0 1.0 1.0 1.0 1.0 .20 | 19/20 |
| route-mix | 0 | 0.49 -> 0.78 | 0/20 | 0/20 | all 0 | 0/20 |
| route-mix | 1 | 0.49 -> 0.84 | 0/20 | 0/20 | all 0 | 0/20 |

- By the pass criterion (11 of 20 on the exact route at the last attack step) no run
  passes.
- `route` seed 1 puts all 20 victims on the exact route for attack steps 9-14 and loses
  16 of them at step 15: the last perturbations leave route cell 29 at altitude 8.1
  below cell 35 at 10, so that move becomes "most likely stays put" (15 victims break
  there with the right action). Its final map has the whole perimeter at 10, like the
  hand-set maps. `route` seed 0 reaches 12/20 at step 7 and then decays.
- `route-mix` never reaches the route on either seed.
- With the old reward and the same settings (round 13 `plain`): 0/20 and 2/20 victims
  were ever on the route at any step.
- Open: the attacker reaches the route but does not hold it to the end of the episode;
  the score reads only the last step.

### Round 15: hold rule and stricter route reward, `2026-11-17` (commit `f5e7a16`)

- `hold_threshold` (default None): while the attacker's own reward of the previous step
  is at or above it, the step's perturbation is replaced by zeros
  (`envs/attack_environment.py`); the attack resumes if the reward falls below.
- The route reward now also ends its walk where the target action most often leaves
  the victim in place, so 1.0 means "seen following the target from start to end
  without getting stuck". Reason: with the first version, holding at reward 1.0 froze
  maps on which victims chose every route action correctly but one move was a steep
  climb (round-14 `route` seed 1 attacker with hold: 0/20 on the exact route).
- Check on the saved round-14 `route` attackers (trained with the first version of the
  reward), 20 victims, exact route at the last step, hold off -> hold at 1.0:
  seed 1: 4/20 -> 20/20 (share on route over steps 9-15: .35 .80 .90 .95 .95 1.0 1.0);
  seed 0: 1/20 -> 0/20.
- Test submitted: `sweep_route_reward.sh` tasks 0-7, experiment `route-reward-v2`:
  6x6, seeds 0-3, `route` and `route-hold` (hold_threshold=1.0).
- **Result (job `7729450`, 6 of 8 scored; no-attack baseline 0/20, target match 0.49):
  no run passes.** Exact route at the last step / target match / best average episode
  return in training (max 15):
  - `route` seed 2: 0/20, 0.54, 3.14; seed 3: 0/20, 0.59, 2.98; seeds 0 and 1 still
    training after 1 h 56 min (at episode ~7,150 of 12,000, best return 3.08 and 2.75).
  - `route-hold` seed 0: 0/20, 0.52, 4.00; seed 1: 2/20, 0.83, 10.25; seed 2: 0/20,
    0.72, 3.29; seed 3: 0/20, 0.92, 5.92.
  - Five of six attackers never averaged more than 6 of 15 in training: with the
    stricter reward they mostly did not learn to get victims along the route at all.
    Round 14 (first version of the reward, no hold) reached target match 0.87 and 1.00.
  - `scripts/diagnose_grid_route.py` on the 6 scored runs (20 victims each): 4 runs
    have no victim on the exact route at any attack step (`route` seeds 2, 3;
    `route-hold` seeds 0, 2), with 4 to 7 wrong route actions per victim on average
    (round 14 `route` seed 1: 19/20 victims had none). `route-hold` seed 3: 8/20 on
    route at some step, 0 at the last. `route-hold` seed 1: 18/20 at some step, share
    on route per step peaks at 0.70 (step 6) and falls to 0.10 by step 15, so the
    hold did not keep the route there.

### Round 16: hold trigger separate from the reward, `2026-11-17` (commit `c425f30`)

- `reward_mode=route` ignores failed moves again (the round-14 reward, which learned).
  The hold uses its own signal: the route measure with `stop_when_stuck=True`, mean
  over victims, whatever the reward mode (`hold_signal` in the step info).
- Check on the saved round-14 attackers with this code, hold at 1.0 (20 victims, exact
  route at the last step): seed 1 20/20, seed 0 0/20 (10/20 at some step). Same as the
  round-15 check, as expected: those attackers were trained without the hold.
- Test submitted: smoke `7734638`, then `sweep_route_reward.sh` tasks 0-7, experiment
  `route-reward-v3` (6x6, seeds 0-3, `route` and `route-hold`). Real job `7734669`.
- **Result (job `7734669`, all 8 scored, about 3.5-3.9 h per run; baseline 0/20, target
  match 0.49): 1 of 8 passes.** Exact route at the last step / at any step / target
  match / best average training return (max 15):
  - `route` seed 0: 0/20, 0/20, 0.75, 3.74; seed 1: 0/20, 0/20, 0.62, 3.28;
    seed 2: 0/20, 0/20, 0.67, 3.86; seed 3: 0/20, 0/20, 0.57, 3.88.
  - `route-hold` seed 0: 0/20, 18/20, 0.74, 9.10; **seed 1: 19/20, 20/20, 1.00, 11.61**;
    seed 2: 0/20, 0/20, 0.71, 3.98; seed 3: 0/20, 0/20, 0.83, 6.10.
  - `route-hold` seed 1 holds the route: share on route per step from step 4:
    .65 .90 .95 .95 .90 .90 1.0 .95 .95 1.0 .95 .95. `route-hold` seed 0 reaches it
    (0.70 at step 5) and loses it (0 from step 12); 17 of 20 first breaks are a wrong
    action, 13 of them in the start state.
  - The four `route` runs use the round-14 reward (the function is equivalent) and the
    round-14 settings, but seeds 0 and 1 did not repeat round 14 (target match 0.87 and
    1.00, 16/20 and 20/20 on route at some step): now 0.75 and 0.62 and never on
    route. Same seed, same settings, different outcome; cause not checked (training
    is not reproducible across jobs, or something else changed between the commits).
  - Round 12 leftovers (job `7710630`, 6x6 own-actions, old reward, 14 h each):
    4/20, 0/20, 0/20 (target match 0.93, 0.83, 0.89).
- **Why the same seed did not repeat (`2026-11-18`): runs were not reproducible.**
  Round 14 and round 16 `route` seed 0 differ from episode 1 in their logs (reward
  0.53 vs 0.67). Locally, two runs of the same command and seed at the same commit also
  differ from episode 1, while their push episodes are identical. Cause: the warm-up
  episodes take `attack_env.action_space.sample()`, and the space's own generator was
  never seeded (not affected by `np.random.seed`). Fix: `main.py` seeds it with
  `cfg["seed"]`. Check: two local runs of 60 episodes (6x6, round-16 `route` settings,
  seed 0) now print identical log lines. Ruled out on the way: hash randomisation,
  thread count, the numba victims, the second Sacred run.
  Consequence: **every result before this fix is one draw of a random warm-up, not "the
  result of seed N"**; differences between configs on the same seed (rounds 12-16) are
  not paired comparisons. Reproducibility on killarney is not checked yet.
- Rerun with the fix submitted `2026-11-18` (commit `4485cd6`): smoke `7751090`
  (tasks 0, 4) plus `7751093` (task 0 again, experiment `smokeb`) to check that the
  same seed repeats on killarney; then `sweep_route_reward.sh` tasks 0-7, experiment
  `route-reward-v4` (same settings as v3).
- **Seeding fix confirmed on killarney**: the two copies of smoke task 0 gave identical
  training logs (100 of 100 lines).
- **Result of `route-reward-v4`** (job `7751186`, about 1 h 10 min per run, 20 fresh
  victims, no-attack baseline 0/20 and target match 0.49): **all 8 runs 0/20.** Target
  match / best average return: `route` seeds 0-3: 0.87 / 4.65, 0.46 / 3.87, 0.69 / 3.80,
  0.49 / 2.80; `route-hold` seeds 0-3: 0.67 / 4.55, 0.64 / 4.80, 0.69 / 3.80,
  0.49 / 2.80. `scripts/diagnose_grid_route.py` was not run on these checkpoints.
- **Why the scores differ between rounds 14, 16 and v4** (checked 2026-11-18):
  - The code path of the `route` reward without hold is unchanged since round 14
    (`git diff 2946482 HEAD`: only the hold rule and the seeding line).
  - For a given seed the exploration (push) episodes are identical in all three rounds;
    only the learned attacker's episodes differ. Last 2,000 training episodes, mean
    episode reward of push episodes (count ending with victims on the route) vs the
    attacker's own episodes in round 14 / round 16 / v4:
    seed 0: 10.11 (375 of 807) vs 2.17 / 2.06 / 2.32; seed 1: 5.74 (0 of 819) vs
    3.64 / 1.41 / 2.42; seed 2: 8.83 (283 of 798) vs - / 0.72 / 0.64; seed 3: 4.72
    (18 of 785) vs - / 2.30 / 1.72.
  - In the runs that scored well the attacker's reward rose late (round 14 seed 0:
    best average 3.03 at episode 8,000, 7.44 at 10,000; round 16 `route-hold` seed 1:
    3.29 at 10,000, 11.61 at 12,000). No v4 run shows that rise.
  - Not checked: why the learned attacker does not reproduce the push episodes.
- **Scoring the exploration search on its own** (commit `503982b`): `main.py` saves the
  search's probabilities (`push_probs.npy`) at every checkpoint;
  `score_attack.py --policy push` plays the most likely direction per dimension at every
  step without noise, `--policy push_sample` draws directions once per victim and adds
  the exploration noise (0.2) as in training. `train_and_score` prints both as extra
  SCORE lines. Submitted as experiment `route-reward-v5` (same 8 runs as v4; smoke
  `7752302` first).
- **Result of `route-reward-v5`** (job `7752361`, 6x6, own-actions observation, 20
  fresh victims, no-attack baseline 0/20 and target match 0.49). Exact route at the
  last step (target match):

  | Config | Seed | Learned attacker | `push` | `push_sample` |
  |---|---|---|---|---|
  | route | 0 | 0/20 (0.87) | 20/20 (1.00) | 8/20 (0.95) |
  | route | 1 | 0/20 (0.46) | 0/20 (0.94) | 0/20 (0.89) |
  | route | 2 | 0/20 (0.69) | 20/20 (1.00) | 6/20 (0.95) |
  | route | 3 | 0/20 (0.49) | 3/20 (0.92) | 4/20 (0.89) |
  | route-hold | 0 | 0/20 (0.67) | 20/20 (1.00) | 13/20 (0.95) |
  | route-hold | 1 | 0/20 (0.64) | 0/20 (0.94) | 0/20 (0.89) |
  | route-hold | 2 | 0/20 (0.69) | 20/20 (1.00) | 9/20 (0.92) |
  | route-hold | 3 | 0/20 (0.49) | 4/20 (0.91) | 4/20 (0.88) |

  - The learned attacker's scores equal v4's on all 8 runs (same seeds, separate jobs).
  - `push` passes the mark (11 of 20) in 4 of 8 runs: seeds 0 and 2, with and without
    hold. Seeds 1 and 3 fail with every policy.
  - `push` is one fixed perturbation repeated at every step: it does not read the
    victim's behaviour at attack time. The search that found it used only the attacker's
    own reward.
  - Not checked: whether the hold ever triggered.
- **Where the `push` policy leaves the route** (`scripts/diagnose_grid_route.py <folder>
  20 policy=push`, the 4 `route` runs of v5, 20 victims each; commit after `ef83d13`):
  - Seeds 0 and 2: 20/20 on the route from attack step 5-6 to step 15, no wrong route
    action in any victim.
  - Seed 1: 0/20 at every step. First break at route position 9 (cell 29, the last
    cell of the right column, where the target turns into the bottom row) in 15
    victims (wrong action there in 13), at position 0-1 in 5 (right action, move most
    likely fails).
  - Seed 3: 3/20 at the last step, 19/20 at some step, share on route per step between
    0.10 and 0.40. First break at positions 7-9 (cells 17, 23, 29), all wrong actions;
    cell 29 wrong in 14 victims.
  - Final maps: all four raise the top row to 10. Seeds 0 and 2 keep the bottom row
    low (2 or 0, corner 35 at 10 or 2); seeds 1 and 3 raise the whole bottom row to 10.
    From cell 29 the goal (30) is 6 moves away both along the bottom row (the target)
    and along the row above it.
- Round 15 final (job `7729450`): `route` seeds 0 and 1 finished after 3 h 20 min:
  0/20 (target match 0.71) and 0/20 (0.48). All 8 round-15 runs fail the pass mark.
- Job `7734669` after 1 h 58 min: all 8 at episode ~6,200-7,000 of 12,000 (about 3.5 h
  per run, slower than the ~1 h 10 min of round-15 tasks 2-7; cause not checked),
  best average return so far 2.3 to 3.9 of 15.

**Round 12 result so far** (job `7710630`, old reward, slow code, 13 of 16 scored;
exact route per seed): 5x5 env view 0, 16, 18, 0; 5x5 own-actions 13, 0, 5, 0;
6x6 env view 4, 0, 20, 0; 6x6 own-actions 0 and three still running after 6 h.

**Lander smoke `7712063`**: 4 attack episodes (K=60, I=10, 3 DQN victims) took
57 min 58 s of training, about 14.5 min per attack episode. The 150-episode
verification in `verify_lander.sh` would need about 36 h per run, over the 24 h limit;
not submitted. The lander needs a speed-up or a smaller budget first.

### Round 17: attacker starts from the search, search gets more budget (`2026-11-18`, commit `5a82636`)

User request: "fix it, make the attacker work".

What the v5 training logs show (push episodes vs the attacker's own, mean episode
reward per 1,000 episodes):

- The search is slow and was still improving when the runs ended. Push mean reward at
  episodes 0-1k / 6-7k / 11-12k: seed 0: 1.0 / 4.4 / 10.1; seed 1: 0.9 / 3.1 / 6.1;
  seed 2: 0.9 / 3.4 / 9.0; seed 3: 0.7 / 3.6 / 5.5. Seed 3 sat at 3.6-3.7 from episode
  3,000 to 10,000 (best single push 4.3) and only then moved; seeds 0 and 2 put their
  first victims on the route at 7-8k.
- The attacker's own episodes stay at 0.5-2.8 for the whole run on every seed and never
  end with victims on the route: the actor does not learn this on its own.

Changes and test (`scripts/slurm/sweep_search_base.sh`, experiment `search-base`,
6x6, route reward, own-actions observation, 12,000 episodes, seeds 0-3, all with
`push_base=True`: the attacker acts as the search's best directions + the actor's
correction):

| Config | Push episodes | Search |
|---|---|---|
| `base` | 40% | as in v5 |
| `base-search` | 80% | as in v5 |
| `base-search-wide` | 80% | `cem_patience=5`: floor doubles (up to 4x) when the elite reward has not risen for 5 updates, resets when it rises |
| `base-search-fast` | 80% | batch 16, elite 4 (twice as many updates) |

Submitted: smoke `7761909` (tasks 0, 8, 12; 150 episodes), then tasks 0-15.

### Diagnosis of the sweep from its training logs (`2026-11-17`)

Means over the last 400 logged episodes of each run (jobs `4423670`), split into the
actor's own episodes and push episodes. "Return" is the attack episode return (max 15).

| Run | Actor return | Push return | Push true target match | Push exact route | Best 50-episode actor average |
|---|---|---|---|---|---|
| 4x4 Q-learning seed 2 | 12.99 | 12.95 | 0.88 | 0.38 | 13.66 |
| 5x5 Q-learning seed 1 | 6.63 | 12.82 | 0.95 | 0.44 | 11.82 |
| 6x6 Q-learning seed 1 | 7.30 | 12.59 | 0.88 | 0.00 | 10.06 |

- On 5x5 and 6x6 the push search finds good perturbations, but the actor ends training
  at the no-attack level (about 7) after an early peak (6x6 seed 1 reached 10.0 by
  episode ~1200 and never improved). The bottleneck is the actor-critic not learning
  and keeping what the search found, not exploration. On 4x4 the actor matches the search.
- Observed reward and true target match agree within 0.05 in every run checked, so the
  behaviour trace is a faithful signal.
- REINFORCE victims: actor return 5.5 (grid) and 9.4 (ring), at or below the no-attack
  level; un-attacked ring-8 victims take the target direction 6/20 times, i.e. they
  have not settled on either route. Like the lander, the victim is not a competent
  learner within the budget, so the cell does not test the attack.
- The best hand-set map gives 0.65 exact route on 6x6 (belief 5 below), so 80% on 6x6
  is above the best known ceiling under the current budget.
- Lander settings differ from the Spiking Pitch Black code (K=60, I=15, TD3,
  failure levels in the observation, 1000-step trajectories, 300-episode encoder
  warm-up); ours used K=20, I=10, DDPG, own-actions observation, 200-step windows.

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
