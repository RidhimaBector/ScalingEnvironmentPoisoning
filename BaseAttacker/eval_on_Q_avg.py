"""Evaluation script for trained attack policies.

Loads a trained DDPG attack policy and evaluates it against fresh victims.
Produces CSV files and plots of accuracy and effort metrics.

Usage:
    python eval_on_Q_avg.py <policy_number>
    python eval_on_Q_avg.py 1000  # Evaluates storage/R_0818/good_model_1000
"""

import os
import sys
import time

import numpy as np
import torch
from gymnasium import spaces

from attack.DDPG import DDPG
from ae.victim_encoder import VictimEncoder
from ae.observation_encoder import ObservationEncoder
from ae.action_translator import ActionTranslator
from agent.attack_dispatch import AttackDispatch
from agent.victim_system import VictimSystem, PrivacyMode
from envs.attack_environment import AttackEnvironment
from envs.env3D_4x4 import Grid3D
from envs.target_def import create_target_policy
from victim.victim_Q import VictimQLearning
from utils.utils_attack import Attack_Done_Identify, Attack_Effort
from yacs.config import CfgNode as CN

# Load config
yaml_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_default.yaml")
with open(yaml_name) as fcfg:
    config = CN.load_cfg(fcfg)
config.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---

policy_no = str(sys.argv[1])
PATH = f"storage/R_0818/good_model_{policy_no}"

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# State dim: derived at eval time from a scratch population (must match training)
nS, nA = 16, 4
action_dim = nS  # 16

_ref_env = Grid3D()
_ref_algo = VictimQLearning(_ref_env.nS, _ref_env.nA, **config.VICTIM.DEFAULT_KWARGS)
_ref_encoder = VictimEncoder.from_population(
    type('_P', (), {
        'algorithm': _ref_algo,
        'env': _ref_env,
        'nS': _ref_env.nS,
        'num_victims': 1,
    })()
)
state_dim = _ref_encoder.embedding_dim

attack_kwargs = dict(config.ATTACK.DEFAULT_KWARGS)
attack_kwargs.update({
    "nb_states": state_dim,
    "nb_actions": action_dim,
    "max_action": 1.0,
    "tau": 0.005,
    "discount": 0.9,
    "eps_greedy_start_episodes": 30,
})
Policy = DDPG(**attack_kwargs)
Policy.load(PATH)

# Evaluation parameters
Tmax = 15
n_victim_pop = 5
Model = "Accuracy-FD_0.90"
Plot_Seed = "Same"


def eval_policy(policy):
    """Evaluate an attack policy against a fresh victim.

    Creates a new VictimSystem and AttackEnvironment, runs the attack
    for Tmax timesteps using the loaded policy.

    Args:
        policy: Trained DDPG attack policy.

    Returns:
        Dict with per-timestep metrics.
    """
    victim_env = Grid3D()
    victim_env.seed(seed)
    victim_algo = VictimQLearning(victim_env.nS, victim_env.nA, **config.VICTIM.DEFAULT_KWARGS)

    population = VictimSystem(
        env=victim_env,
        algorithm=victim_algo,
        config=config,
        privacy_mode=PrivacyMode.FULL_WHITEBOX,
    )

    attack_env = AttackEnvironment(
        victim_population=population,
        attack_dispatch=AttackDispatch(),
        victim_train_episodes=80,
        config=config,
    )

    # Build encoder services (no Sacred tracker — fallback to VictimSystem)
    obs_encoder = ObservationEncoder(VictimEncoder.from_population(population))
    action_translator = ActionTranslator.from_space(
        spaces.Box(-1.0, 1.0, shape=(population.nS,), dtype=np.float64)
    )

    target = create_target_policy(victim_env.nS, victim_env.nA, path_type="Mp")

    attack_env.reset()
    obs = obs_encoder.get_initial_embedding()

    metrics_per_timestep = {
        'accuracy': [],
        'accuracy_sftmx': [],
        'accuracy_sftmx_complete': [],
        'effort': [],
        'time': [],
    }

    prev_dynamics = population.get_env_dynamics().flatten()

    for t in range(Tmax):
        tic = time.time()

        raw_action = policy.select_on_policy_action(np.array(obs))
        dispatch_action = action_translator.translate(raw_action)
        _, reward, terminated, truncated, step_info = attack_env.step(dispatch_action)
        next_obs = obs_encoder.encode(step_info['victim_results'])

        done, acc, acc_sftmx, acc_sftmx_complete = Attack_Done_Identify(
            target.copy(), victim_algo.Q
        )
        current_dynamics = population.get_env_dynamics().flatten()
        effort = float(np.mean(np.abs(current_dynamics - prev_dynamics)))

        metrics_per_timestep['accuracy'].append(acc)
        metrics_per_timestep['accuracy_sftmx'].append(acc_sftmx)
        metrics_per_timestep['accuracy_sftmx_complete'].append(acc_sftmx_complete)
        metrics_per_timestep['effort'].append(effort)
        metrics_per_timestep['time'].append(time.time() - tic)

        prev_dynamics = current_dynamics
        obs = next_obs

    return metrics_per_timestep


# --- Run evaluations ---

all_results = []
for i_pop in range(n_victim_pop):
    print(f"Evaluating victim population {i_pop + 1}/{n_victim_pop}")
    result = eval_policy(Policy)
    all_results.append(result)

# --- Build DataFrames and plots ---
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.8)
sns.set_style("whitegrid")

atk_time = np.arange(1, Tmax + 1)
vic_Pop_names = [f"VP{i + 1}" for i in range(n_victim_pop)]

rows = []
for i_pop in range(n_victim_pop):
    for t in range(Tmax):
        rows.append({
            'Attacker Timestep': atk_time[t],
            'Model': Model,
            'Accuracy': all_results[i_pop]['accuracy'][t],
            'Sftmx Accuracy': all_results[i_pop]['accuracy_sftmx'][t],
            'Effort': all_results[i_pop]['effort'][t],
            'Time': all_results[i_pop]['time'][t],
            'Victim Pop': vic_Pop_names[i_pop],
            'Seed': Plot_Seed,
        })

df = pd.DataFrame(rows)
df.to_csv(f"{policy_no}_eval_results_{Model}_Seed{Plot_Seed}.csv", index=False)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
sns.lineplot(x="Attacker Timestep", y="Accuracy", hue="Model", data=df, ax=axs[0, 0])
sns.lineplot(x="Attacker Timestep", y="Sftmx Accuracy", hue="Model", data=df, ax=axs[0, 1])
sns.lineplot(x="Attacker Timestep", y="Effort", hue="Model", data=df, ax=axs[1, 0])
sns.lineplot(x="Attacker Timestep", y="Time", hue="Model", data=df, ax=axs[1, 1])
plt.tight_layout()
plt.savefig(f"{policy_no}_eval_{Model}_Seed{Plot_Seed}.png")
print(f"Results saved to {policy_no}_eval_results_{Model}_Seed{Plot_Seed}.csv")
