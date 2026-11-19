"""DDPG attacker (with optional TD3 extensions and elite imitation).

The attacker's policy maps its observation to a perturbation vector. It is trained
off-policy from a replay buffer of attack steps.

Options beyond plain DDPG, all off by default:
  td3        twin critics (minimum of two targets), target-policy smoothing, and
             delayed actor updates
  bc_weight  the actor is also pulled towards the actions of the best episodes seen
             so far (set_elite)
"""

import copy
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from attack.random_process import OrnsteinUhlenbeckProcess
from attack.replay_buffer import ReplayBuffer
from core.attack_algorithm import AttackAlgorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(array) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)


def _fanin_init(size):
    bound = 1.0 / np.sqrt(size[0])
    return torch.empty(size).uniform_(-bound, bound)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.lerp_(s, tau)  # t <- (1 - tau) * t + tau * s


class Actor(nn.Module):
    """observation -> perturbation in [-max_action, max_action]."""

    def __init__(
        self, nb_states, nb_actions, max_action, hidden1=400, hidden2=300, init_w=3e-3
    ):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.fc1.weight.data = _fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = _fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """(observation, perturbation) -> expected discounted attack reward."""

    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.fc1.weight.data = _fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = _fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], 1)))
        return self.fc3(x)


class DDPG(AttackAlgorithm):
    """See the module docstring. Hyperparameters come from config/config_default.yaml
    (ATTACK.DEFAULT_KWARGS) and defaults.py."""

    def __init__(
        self,
        nb_states,
        nb_actions,
        max_action,
        hidden1=400,
        hidden2=300,
        init_w=3e-3,
        prate=1e-4,
        rate=1e-3,
        ou_theta=0.15,
        ou_mu=0.0,
        ou_sigma=0.2,
        bsize=256,
        tau=0.005,
        discount=0.9,
        epsilon_divisor=1000,
        eps_greedy_start_episodes=30,
        updates_per_step=1,
        expl_noise=0.0,
        td3=False,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        bc_weight=0.0,
        correction_scale=1.0,
        correct_held=True,
        **_unused,
    ):
        self.nb_states, self.nb_actions, self.max_action = (
            nb_states,
            nb_actions,
            max_action,
        )
        net = dict(hidden1=hidden1, hidden2=hidden2, init_w=init_w)

        self.actor = Actor(nb_states, nb_actions, max_action, **net).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=prate)
        self.critic = Critic(nb_states, nb_actions, **net).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=rate)

        self.td3 = td3
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq if td3 else 1
        if td3:
            self.critic2 = Critic(nb_states, nb_actions, **net).to(device)
            self.critic2_target = copy.deepcopy(self.critic2)
            self.critic2_optim = Adam(self.critic2.parameters(), lr=rate)

        self.bc_weight = bc_weight
        self._elite_s = self._elite_a = None
        # Base action from the push search; the actor outputs a correction to it.
        # All zeros (the default) gives plain DDPG.
        self._base = np.zeros(nb_actions, dtype=np.float32)
        # The correction is the actor's output times this, so below 1 the actor can
        # move each dimension at most that far from the base.
        self.correction_scale = correction_scale
        # False: no correction on dimensions where the base is 0 (the search holds them)
        self.correct_held = correct_held

        self._buffer = ReplayBuffer(state_dim=nb_states, action_dim=nb_actions)
        self._ou_noise = OrnsteinUhlenbeckProcess(
            size=nb_actions, theta=ou_theta, mu=ou_mu, sigma=ou_sigma
        )
        self.batch_size, self.tau, self.discount = bsize, tau, discount
        self.updates_per_step = updates_per_step
        self.expl_noise = expl_noise  # constant Gaussian exploration noise
        self.epsilon, self._depsilon = (
            1.0,
            1.0 / epsilon_divisor,
        )  # decaying OU noise scale
        self._warmup_episodes = eps_greedy_start_episodes
        self._n_updates = 0
        self._training = True

    # --- acting ---

    def act_greedy(self, state: np.ndarray) -> np.ndarray:
        """The policy's action without exploration noise."""
        with torch.no_grad():
            action = self._policy(self.actor, _tensor(state).reshape(1, -1))
            return action.cpu().numpy().squeeze(0)

    def _policy(self, actor: nn.Module, state: torch.Tensor) -> torch.Tensor:
        """base + the actor's correction, kept inside the action range."""
        base = _tensor(self._base)
        correction = self.correction_scale * actor(state)
        if not self.correct_held:
            correction = correction * (base != 0)
        return (correction + base).clamp(-1.0, 1.0)

    def set_base(self, base: np.ndarray) -> None:
        self._base = np.asarray(base, dtype=np.float32).reshape(-1)

    def clear_buffer(self) -> None:
        self._buffer = ReplayBuffer(state_dim=self.nb_states, action_dim=self.nb_actions)

    def act(self, state: np.ndarray) -> np.ndarray:
        """The policy's action with exploration noise (training)."""
        action = self.act_greedy(state)
        if self._training:
            action = action + max(self.epsilon, 0) * self._ou_noise.sample()
            if self.expl_noise > 0:
                action = action + np.random.normal(
                    0.0, self.expl_noise, size=action.shape
                )
            self.epsilon = max(self.epsilon - self._depsilon, 0.01)
        return np.clip(action, -1.0, 1.0)

    # --- learning ---

    def store_transition(self, obs, action, next_obs, reward, done) -> None:
        self._buffer.add(obs, action, next_obs, reward, done)

    def set_elite(self, states, actions) -> None:
        """Transitions of the best episodes so far, for the imitation term."""
        self._elite_s = np.asarray(states, dtype=np.float32)
        self._elite_a = np.asarray(actions, dtype=np.float32)

    def ready_to_train(self) -> bool:
        return len(self._buffer) >= self.batch_size

    @property
    def warmup_episodes(self) -> int:
        return self._warmup_episodes

    def update(self, episode: int = 0) -> Dict[str, float]:
        """`updates_per_step` gradient steps on batches from the replay buffer."""
        critic_loss_sum = q_sum = 0.0
        for _ in range(self.updates_per_step):
            state, action, next_state, reward, not_done = (
                _tensor(x) for x in self._buffer.sample(self.batch_size)
            )
            self._n_updates += 1

            with torch.no_grad():
                next_action = self._policy(self.actor_target, next_state)
                if self.td3:
                    smoothing = (
                        torch.randn_like(next_action) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_action = (next_action + smoothing).clamp(
                        -self.max_action, self.max_action
                    )
                next_q = self.critic_target(next_state, next_action)
                if self.td3:
                    next_q = torch.min(
                        next_q, self.critic2_target(next_state, next_action)
                    )
                target_q = reward + self.discount * not_done * next_q

            critics = [(self.critic, self.critic_optim)]
            if self.td3:
                critics.append((self.critic2, self.critic2_optim))
            for critic, optim in critics:
                loss = nn.functional.mse_loss(critic(state, action), target_q)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optim.step()
                if critic is self.critic:
                    critic_loss_sum += loss.item()

            if self._n_updates % self.policy_freq == 0:
                q = self.critic(state, self._policy(self.actor, state)).mean()
                actor_loss = -q
                if self.bc_weight > 0 and self._elite_s is not None:
                    idx = np.random.randint(0, len(self._elite_s), size=self.batch_size)
                    imitation = (
                        (
                            self.actor(_tensor(self._elite_s[idx]))
                            - _tensor(self._elite_a[idx])
                        )
                        ** 2
                    ).mean()
                    actor_loss = actor_loss + self.bc_weight * imitation
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optim.step()
                q_sum += q.item() * self.policy_freq

                _soft_update(self.actor_target, self.actor, self.tau)
                _soft_update(self.critic_target, self.critic, self.tau)
                if self.td3:
                    _soft_update(self.critic2_target, self.critic2, self.tau)

        n = self.updates_per_step
        return {"critic_loss": critic_loss_sum / n, "actor_loss": q_sum / n}

    # --- persistence ---

    def save(self, path: str) -> None:
        torch.save(self.critic.state_dict(), path + "_critic")
        torch.save(self.critic_optim.state_dict(), path + "_critic_optimizer")
        torch.save(self.actor.state_dict(), path + "_actor")
        torch.save(self.actor_optim.state_dict(), path + "_actor_optimizer")
        if self.td3:
            torch.save(self.critic2.state_dict(), path + "_critic2")
        np.save(path + "_base.npy", self._base)

    def load(self, path: str) -> None:
        self.critic.load_state_dict(torch.load(path + "_critic", map_location=device))
        self.critic_optim.load_state_dict(
            torch.load(path + "_critic_optimizer", map_location=device)
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(path + "_actor", map_location=device))
        self.actor_optim.load_state_dict(
            torch.load(path + "_actor_optimizer", map_location=device)
        )
        self.actor_target = copy.deepcopy(self.actor)
        if os.path.exists(path + "_base.npy"):  # absent in checkpoints made before push_base
            self._base = np.load(path + "_base.npy").astype(np.float32)
        if self.td3 and os.path.exists(path + "_critic2"):
            self.critic2.load_state_dict(
                torch.load(path + "_critic2", map_location=device)
            )
            self.critic2_target = copy.deepcopy(self.critic2)

    def eval(self) -> None:
        """Switch off exploration noise."""
        self._training = False
        self.actor.eval()
