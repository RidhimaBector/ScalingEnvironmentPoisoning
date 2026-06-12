"""Canonical DDPG implementation for comparison with the project's DDPG.py.

This follows the reference implementation from:
  Lillicrap et al. (2015) - https://arxiv.org/abs/1509.02971
  Fujimoto et al. TD3 paper DDPG baseline - https://arxiv.org/abs/1802.09477

Key differences from DDPG.py (annotated with [DIFF]):
  1. [DIFF-1] updates_per_step=1 (not hardcoded 15). High update-to-data ratio
     is the primary driver of Q-value explosion visible in training plots.
  2. [DIFF-2] Actor gradient clipping added (critic already had it in DDPG.py).
     Unconstrained actor gradients let policy loss diverge after Q explodes.
  3. [DIFF-3] Gaussian noise (σ=0.1) instead of OU process. Simpler, equally
     effective in practice; OU is not wrong but adds unnecessary state.
  4. [DIFF-4] actor_loss logged as +Q_mean (positive), not the raw negated loss,
     so the plot shows "mean Q value the actor achieves" — easier to interpret.

NOTE: Actor loss = -Q(s, μ(s)).mean() is NEGATIVE BY DESIGN when Q > 0.
The existing DDPG.py logs the raw negated value, making it look alarming.
This file logs +Q_mean so you can see directly whether the actor is improving.
"""

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from core.attack_algorithm import AttackAlgorithm
from attack.util import to_tensor, to_numpy, soft_update, hard_update, USE_CUDA
from utils.utils_buf import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class _Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, max_action, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_action = max_action
        self.fc1.weight.data = _fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = _fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        return self.max_action * self.tanh(self.fc3(out))


class _Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.fc1.weight.data = _fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = _fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(torch.cat([out, action], dim=1)))
        return self.fc3(out)


class DDPGCanonical(AttackAlgorithm):
    """Canonical DDPG — use this to isolate instability from DDPG.py.

    Plug-in replacement for DDPG in the attack framework (same ABC).
    Set updates_per_step=15 to replicate the original; the default of 1
    is the standard ratio from the paper.
    """

    def __init__(
        self,
        seed,
        nb_states,
        nb_actions,
        max_action,
        hidden1=400,
        hidden2=300,
        init_w=3e-3,
        prate=1e-4,          # actor lr
        rate=1e-3,           # critic lr
        bsize=64,
        tau=0.005,
        discount=0.99,
        # [DIFF-1] canonical default is 1 update per env step
        updates_per_step=1,
        # [DIFF-3] Gaussian noise std; set to 0 for deterministic eval
        expl_noise=0.1,
        eps_greedy_start_episodes=0,
        is_training=True,
        **kwargs,
    ):
        if seed > 0:
            torch.manual_seed(seed)
            if USE_CUDA:
                torch.cuda.manual_seed(seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.max_action = max_action

        net_cfg = dict(hidden1=hidden1, hidden2=hidden2, init_w=init_w)
        self.actor = _Actor(nb_states, nb_actions, max_action, **net_cfg)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=prate)

        self.critic = _Critic(nb_states, nb_actions, **net_cfg)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=rate)

        self._buffer = ReplayBuffer(state_dim=nb_states, action_dim=nb_actions)

        self.batch_size = bsize
        self.tau = tau
        self.discount = discount
        self.updates_per_step = updates_per_step  # [DIFF-1]
        self.expl_noise = expl_noise              # [DIFF-3]
        self.eps_greedy_start_episodes = eps_greedy_start_episodes
        self.is_training = is_training

        self._loss_log: List[list] = []

        if USE_CUDA:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

    # ------------------------------------------------------------------
    # AttackAlgorithm ABC
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> np.ndarray:
        """Select action with Gaussian exploration noise. [DIFF-3]"""
        action = to_numpy(self.actor(to_tensor(state.reshape(1, -1)))).squeeze(0)
        if self.is_training and self.expl_noise > 0:
            noise = np.random.normal(0, self.expl_noise, size=self.nb_actions)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    def store_transition(self, obs, action, next_obs, reward, done):
        self._buffer.add(obs, action, next_obs, reward, done)

    def ready_to_train(self) -> bool:
        return len(self._buffer) >= self.batch_size

    def update(self, episode: int = 0) -> Dict[str, float]:
        """[DIFF-1] Run `updates_per_step` gradient steps (default=1)."""
        loss_critic_sum = 0.0
        loss_actor_sum = 0.0

        for _ in range(self.updates_per_step):
            state, action, next_state, reward, not_done = self._buffer.sample(self.batch_size)

            state_t = to_tensor(state)
            action_t = to_tensor(action)
            next_state_t = to_tensor(next_state)
            reward_t = to_tensor(reward)
            not_done_t = to_tensor(not_done)

            # --- Critic update ---
            with torch.no_grad():
                next_action = self.actor_target(next_state_t)
                target_q = reward_t + self.discount * not_done_t * self.critic_target(next_state_t, next_action)

            current_q = self.critic(state_t, action_t)
            critic_loss = nn.functional.mse_loss(current_q, target_q)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()

            # --- Actor update ---
            # policy_loss = -Q(s, μ(s)).mean()  →  this IS negative when Q > 0 (correct)
            # [DIFF-4] we report +Q_mean so the plotted value shows actor's achieved Q
            actor_loss = -self.critic(state_t, self.actor(state_t)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # [DIFF-2]
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            loss_critic_sum += critic_loss.item()
            loss_actor_sum += (-actor_loss.item())  # [DIFF-4] log +Q_mean

        n = self.updates_per_step
        avg_critic = loss_critic_sum / n
        avg_actor_q = loss_actor_sum / n  # positive when actor is doing well

        self._loss_log.append([episode, 0, avg_critic, avg_actor_q])
        return {"critic_loss": avg_critic, "actor_loss": avg_actor_q}

    @property
    def warmup_episodes(self) -> int:
        return self.eps_greedy_start_episodes

    def get_loss_log(self) -> list:
        return self._loss_log

    def save_buffer(self, path: str) -> None:
        self._buffer.saveBuffer(path)

    def save(self, filename: str) -> None:
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optim.state_dict(), filename + "_actor_optimizer")

    def load(self, filename: str) -> None:
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location="cpu"))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer", map_location="cpu"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location="cpu"))
        self.actor_optim.load_state_dict(torch.load(filename + "_actor_optimizer", map_location="cpu"))
        self.actor_target = copy.deepcopy(self.actor)
