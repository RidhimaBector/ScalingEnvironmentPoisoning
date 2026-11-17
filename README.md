# Scaling Environment Poisoning

Black-box **environment-poisoning attacks** on reinforcement learners.

An attacker cannot touch the victim. It can only nudge the victim's *environment* and
watch how the victim behaves. By learning which nudges change the victim's behaviour
in which way, it steers the victim into a behaviour of the attacker's choosing.

This repository builds on *Spiking Pitch Black: Poisoning an Unknown Environment to
Attack Unknown Reinforcement Learners* (AAMAS 2022).

It is written for two uses:

1. **Reproducing** the experiments (grid world, ring, LunarLander).
2. **Extending** them: plug in your own environment, victim learner, attacker or
   behaviour encoder. The attack pipeline knows nothing about any particular game.

---

## How it works

```
                 one attack episode (fresh victim, unpoisoned environment)
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                  │
        │   attacker observes ──► attacker sends a ──► environment's       │
        │   (behaviour embedding   perturbation        dynamics change     │
        │    + its own actions)                              │             │
        │          ▲                                         ▼             │
        │   attacker is rewarded ◄── victim's behaviour ◄── victim         │
        │   (how close behaviour      is observed            relearns      │
        │    is to the target)                                             │
        │                         repeated for 15 attack steps             │
        └──────────────────────────────────────────────────────────────────┘
                 after each episode the attacker learns from its replay buffer
```

- The **victim** is an ordinary RL agent (Q-learning, SARSA, REINFORCE, DQN, VPG, A2C).
  It is never modified; it learns correctly in the world it is given.
- The **attacker** (DDPG) outputs a perturbation vector. It is not told what the
  numbers mean. It sees an *embedding of the victim's behaviour* and the running sum
  of its own past actions; never the victim's parameters or the environment's rules.
- **Push exploration** makes the search tractable: in some episodes every perturbation
  dimension is driven one way for the whole episode, and the directions that earned
  the most reward are sampled more often afterwards.
- **Evaluation** uses ground truth the attacker never saw: fresh victims, compared
  with and without the attack.

## Install

Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate
pip install -r BaseAttacker/requirements.txt
```

LunarLander additionally needs Box2D (`pip install swig box2d-py`). Everything else
runs without it.

Check the installation (about 10 seconds):

```bash
cd BaseAttacker
python -m pytest tests -q
python main.py with quick_test
```

## Quick start

All commands run from `BaseAttacker/`.

```bash
# Train an attacker on the 4x4 grid against a Q-learning victim (about 35 min, 1 CPU)
python main.py with seed=0 experiment=demo
RUN=results/demo/grid3d-4x4__qlearning__default__seed0

# Score it on 20 fresh victims, against a no-attack baseline
python score_attack.py --run_dir $RUN

# Draw what it did (grid environments)
python plot_grid_attack.py --run_dir $RUN
```

`score_attack.py` prints one line:

```
SCORE env=grid3d victim=qlearning | target_match 0.46 -> 0.99 | route 0/10 -> 9/10 | ...
```

(an actual result, scored with `--episodes 10`)

read as "without the attack → with the attack": `target_match` is the share of target
states where the victim's preferred action is the attacker's; `route` counts victims
whose preferred route is exactly the target route.

Other setups are one argument away:

```bash
python main.py with victim_algo=sarsa
python main.py with grid_size=6 max_episodes=12000
python main.py with env=ring env_args=n_states:12 max_episodes=3000
python main.py with lander                      # LunarLander + DQN victim
python main.py with lander victim_algo=a2c
```

Every setting is listed with its default and meaning in
[`BaseAttacker/defaults.py`](BaseAttacker/defaults.py).

### Run names

Every run is named by one convention ([`naming.py`](BaseAttacker/naming.py)), used for
its results folder and for its path on the Hugging Face Hub:

```
results/<experiment>/<env>-<variant>__<victim>__<config>__seed<N>
results/sweep-3x3x3/grid3d-5x5__sarsa__default__seed0
```

Set the first and third parts with `experiment=<name>` and `config_name=<label>`; the
rest is derived from the settings.

### Experiment tracking

Runs are recorded with [Sacred](https://github.com/IDSIA/sacred): the attacker's run
under `sacred_runs/`, the victims' under `sacred_runs_victims/`. To use MongoDB
instead, set `SACRED_MONGO_URL=localhost:27017` (and optionally `SACRED_DB_NAME`).
Tracking is write-only; nothing in training reads it back.

## Reproducing the experiments

Each sweep is a SLURM array script that trains and then scores every run. Pass your
own allocation; nothing is hard-coded.

```bash
cd BaseAttacker && mkdir -p logs
export VENV=../venv                      # and PYTHON_MODULE=python/3.12 if your cluster needs it

sbatch --account=<acct> --array=0-17 scripts/slurm/sweep_grid4.sh          # 4x4 grid, 18 seeds
sbatch --account=<acct> --array=0-17 scripts/slurm/sweep_ablation_4x4.sh   # one-change ablation
sbatch --account=<acct> --array=0-35 scripts/slurm/sweep_grid_sizes.sh     # 5x5, 6x6, 8x8
sbatch --account=<acct> --array=0-80 scripts/slurm/sweep_3x3x3.sh          # 3 envs x 3 variants x 3 victims

python scripts/collect_scores.py results/sweep-3x3x3/* --by cell           # table of results
```

Per-cluster values and first-time setup on Alliance Canada clusters are in
[`scripts/slurm/README.md`](BaseAttacker/scripts/slurm/README.md).

Without a cluster, each script's `train_and_score` line shows the exact `main.py`
settings of a run; start them one at a time.

| Sweep | Runs | Time per run (1 CPU) |
|---|---|---|
| `sweep_grid4.sh` | 18 | about 35 min |
| `sweep_ablation_4x4.sh` | 18 | about 30 min |
| `sweep_grid_sizes.sh` | 36 | 2 to 5 h |
| `sweep_3x3x3.sh` | 81 | 10 min (ring) to 21 h (LunarLander) |

Things to know when comparing with reported numbers:

- **Seeds matter.** Training is unstable: on the 4x4 grid roughly half of the seeds
  end with an attacker that puts most fresh victims on the exact target route, and the
  rest fail outright. Report all seeds.
- **Model selection without ground truth.** The attacker's own best 50-episode training
  reward (`best_avg_reward` in the log) separates working seeds from failed ones, so
  "train several seeds, keep the highest-reward one" is a legitimate black-box procedure.
- **The exact-route criterion gets stricter with size.** One wrong state breaks it, so
  also read `target_match`.
- Runs are seeded, but results are not bit-identical across machines or library versions.

### Publishing runs

Cluster runs upload themselves when they finish ([`publish.py`](BaseAttacker/publish.py)):
the scored checkpoint, `run_config.json` and `score.json` to a model repository, and
one JSON record per run plus its log to a dataset repository, both under the run's
name. Set `HF_ATTACKERS_REPO` and `HF_RESULTS_REPO` to your own repositories and log
in once with `hf auth login` (or set `HF_TOKEN`). `HF_PUBLISH=0` switches it off. An
upload that fails never fails the run; `python publish.py --pending results` retries.

A trained example is included: `python score_attack.py --run_dir
checkpoints/sweep-3x3x3/grid3d-4x4__qlearning__default__seed2`.

## Extending

The pipeline is built from four interfaces in [`BaseAttacker/core/`](BaseAttacker/core).
Implement one, add a line to [`BaseAttacker/registry.py`](BaseAttacker/registry.py),
and everything else (training, scoring, sweeps) works unchanged.

### Your own environment

Subclass `core.victim_environment.VictimEnvironment` (a gymnasium `Env`).
[`envs/ring.py`](BaseAttacker/envs/ring.py) is a complete example in 90 lines.

| You implement | Meaning |
|---|---|
| `reset()`, `step(action)`, `seed(seed)` | the game the victim plays (gymnasium API) |
| `observation_space`, `action_space` | discrete actions; states discrete or a vector |
| `perturbation_space` | what the attacker may send each step (a `Box`) |
| `apply_perturbation(vector)` | how a perturbation changes the dynamics (cumulative) |
| `get_dynamics()`, `dynamics_dim`, `reset_dynamics()` | current dynamics as a vector; undo all poisoning |
| `make_target_policy()` **or** `target_trajectories()` | the behaviour the attacker wants |
| `oracle_route_match(...)`, `dynamics_bounds()` | optional: ground truth for scoring; bounds |

The target is a per-state table (`make_target_policy`, discrete states) or a recorded
trajectory (`target_trajectories`, any states).

```python
# registry.py
ENV_MAP = _LazyClasses({
    ...
    "mygame": "envs.mygame:MyGame",
})
```

```bash
python -m pytest tests/test_interfaces.py -k mygame   # does it honour the contract?
python -m ae.train_ae --env mygame                    # discrete states: train its behaviour autoencoder
python main.py with env=mygame                        # attack it
```

Pre-training simulates victims in your environment. To avoid that, fit the autoencoder
during the attack on the traces the attacker itself observes (its first 500 episodes
are then exploration only): `python main.py with env=mygame ae_model_path=online`.

For vector-valued states skip the autoencoder and use the trajectory encoder:
`python main.py with env=mygame victim_algo=dqn encoder_mode=lstm reward_mode=embedding`.

### Your own victim learner

Subclass `core.victim_algorithm.VictimAlgorithm`:

| You implement | Meaning |
|---|---|
| `train(env, num_episodes)` | learn for some episodes |
| `act(state)`, `reset()` | choose an action; forget everything |
| `get_trajectories()` | `(prev_state, prev_action, state, action)` tuples since the last call: **this is all a black-box attacker sees** |
| `get_behavior_trace()`, `get_policy_matrix()` | tabular learners: last action per state; the policy table (evaluation only) |
| `greedy_actions_for(states)` | non-tabular learners: preferred actions (evaluation only) |
| `from_env(env, **kwargs)` (classmethod) | optional: build yourself from an environment |

[`victim/victim_qlearning.py`](BaseAttacker/victim/victim_qlearning.py) (tabular) and
[`victim/victim_pg.py`](BaseAttacker/victim/victim_pg.py) (neural) are short examples.
Register it in `VICTIM_ALGO_MAP` and run `python main.py with victim_algo=<name>`.

### Your own attacker or encoder

- **Attacker:** subclass `core.attack_algorithm.AttackAlgorithm` (`act`,
  `store_transition`, `ready_to_train`, `update`, `save`, `load`) and add it to
  `ATTACK_ALGO_MAP`. [`attack/ddpg.py`](BaseAttacker/attack/ddpg.py) is the reference.
- **Encoder:** subclass `core.encoder.Encoder` (`encode`, `embedding_dim`,
  `get_initial_embedding`, `compute_reward`) and add a branch in
  `builders.build_encoders`. An encoder turns what the victims exposed into the
  attacker's observation and reward.

### Using the pieces from your own code

```python
from builders import (resolve_config, build_population, build_encoders,
                      build_attack_env, build_attack_algo)

cfg = resolve_config({"env": "ring", "victim_algo": "sarsa"})
population = build_population(cfg)
obs_encoder, translator = build_encoders(population, cfg)
attack_env = build_attack_env(population, cfg, obs_encoder)   # a gymnasium Env
attacker = build_attack_algo(attack_env, cfg)

obs, _ = attack_env.reset()
obs, reward, _, _, info = attack_env.step(translator.translate(attacker.act(obs)))
```

## Repository layout

```
BaseAttacker/
  main.py               train an attacker
  score_attack.py       score a trained attacker on fresh victims (any environment)
  plot_grid_attack.py   figure of an attack on a grid
  defaults.py           every setting and its default
  builders.py           build the components from settings
  registry.py           names -> environment / victim / attacker classes
  experiment.py         Sacred tracking
  core/                 the four interfaces
  envs/                 grid3d, ring, lunar_lander; attack_environment (the attacker's MDP)
  victim/               Q-learning, SARSA, REINFORCE, DQN, VPG, A2C
  attack/               DDPG attacker, push exploration, replay buffer, PPO (experimental)
  ae/                   behaviour encoders: trace autoencoder, trajectory LSTM, white-box
  agent/                victim population, perturbation dispatch, Sacred trackers
  utils/metrics.py      accuracy and distance measures
  config/               hyperparameters internal to the learners
  naming.py             the run naming convention
  publish.py            upload finished runs to the Hugging Face Hub
  checkpoints/          one trained attacker, as an example
  scripts/              SLURM sweeps, result collection, feasibility probes
  tests/                interface contract tests and end-to-end smoke tests
```

## Limitations

- Victims must have **discrete actions**.
- The trace autoencoder and the trace reward need **discrete states**; for vector
  states use the trajectory encoder and the embedding reward.
- An autoencoder is trained **per environment and size**. Those for the shipped
  environments are in `ae/models/`.
  `ae_model_path=online` needs none.
- `push_base=True` makes the attacker act as the push search's best directions plus
  the actor's correction. Both options are off by default; `scripts/slurm/sweep_push_base.sh`
  tests them.
- The PPO attacker and the white-box encoder run but were not part of the reported
  experiments.

## License

MIT; see [LICENSE](LICENSE). The LunarLander physics is adapted from OpenAI Gym (MIT).

## Citation

```
Spiking Pitch Black: Poisoning an Unknown Environment to Attack Unknown
Reinforcement Learners. AAMAS 2022.
```
