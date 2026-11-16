# Rough idea: an LLM as the black-box attacker

Status: idea only, nothing built. Written 2026-11-16.

## Question

Can an untrained LLM, given only what the current black-box attacker sees, poison the
grid so the victim takes the target route? What score does it get compared with the
trained DDPG attacker?

## Setup (unchanged from the current experiments)

- Environment: Grid3D, N x N, altitude per cell in [0, 10].
- Victim: tabular Q-learning, relearns for 80 episodes after every attack step.
- Attack episode: 15 steps, each step changes every cell's altitude by a value in [-1, 1].
- Target: the perimeter route (top row east, right column south, bottom row west).
- Score: 20 fresh victims, share whose greedy route at step 15 is exactly the target.

Only the attacker changes: the DDPG policy is replaced by an LLM call per attack step.

## Black-box constraint

The LLM gets exactly the information the current attacker gets, as text:

| Given | Not given |
|---|---|
| Grid size, start cell, goal cell | The transition rules (how altitude affects movement) |
| The target route | The victim's Q-table or greedy policy |
| Current altitude of every cell (the attacker controls it) | The victim's algorithm or hyperparameters |
| Victim's last observed action in each state (the behaviour trace) | Ground-truth route match (evaluation only) |
| Its own previous actions and observed rewards in this episode | Anything about earlier experiments or working maps |
| Reward: share of target states whose last observed action matches the target | |

The autoencoder embedding is dropped: it is meaningless to an LLM, and the raw trace is
the same information.

## Loop

1. Reset: fresh victim, natural altitude.
2. For each of 15 steps:
   - Build the prompt: task description, grid drawn as a table, target route, current
     altitudes, behaviour trace, history of (altitude change, reward) so far.
   - LLM returns one number in [-1, 1] per cell (JSON list, N*N values).
   - Apply it, victim relearns for 80 episodes, record trace and reward.
3. Evaluate with the existing `eval_attack.py` logic.

## Conditions to compare

1. **Zero-shot.** One attack episode per victim, no memory between victims. Tests what
   the LLM can infer within 15 steps.
2. **In-context learning.** The LLM plays K attack episodes in a row (say 10) and keeps a
   running summary of what it tried and the rewards. Then it is scored on fresh victims
   using its final strategy. This is the closest analogue to training.
3. **Baselines already in hand.** Trained DDPG with adaptive push (per grid size), and a
   no-attack baseline (natural route).

## What to measure

- Exact-route share and greedy match on 4x4, 5x5, 6x6, 8x8 (same metrics as now).
- Number of environment interactions used: the DDPG attacker needs about 6000 attack
  episodes; the LLM uses 1 (zero-shot) or K.
- Whether it finds a blocking interior on 8x8, where the simple low interior fails.

## Things that could go wrong

- **Leakage.** The prompt must not describe the dynamics or hint at a working map, or
  the result is not black-box. Keep the prompt in the repo and review it for this.
- **No signal early on.** In the first steps the trace reward barely moves, so the LLM
  has little to reason from. Zero-shot may fail for that reason alone.
- **Noisy reward.** The trace reward is a single victim's last actions; the LLM may
  over-read one noisy step.
- **Prior knowledge.** The LLM may guess that lower altitude is easier to move through.
  That is a prior, not leaked information, but it should be stated when reporting.
- **Cost.** 20 victims x 15 steps = 300 calls per grid size per condition; the
  in-context condition adds K x 15 calls. Small, but not free.

## Rough build

- One new class behind the existing attacker interface (`act(obs)` in `main.py` and
  `eval_attack.py`), which formats the prompt and parses the reply.
- A flag in `eval_attack.py` to use it instead of loading a DDPG checkpoint.
- Runs from the laptop; no cluster, no training.
