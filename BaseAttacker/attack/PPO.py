"""PPO attack algorithm for learning environment perturbation policies.

Implements Proximal Policy Optimization with clipped surrogate objective,
Gaussian actor with tanh squashing, V(s) critic, and GAE advantage
estimation. On-policy alternative to DDPG for the attack agent.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from core.attack_algorithm import AttackAlgorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Rollout Buffer (on-policy)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size on-policy rollout storage with GAE computation."""

    def __init__(self, state_dim: int, action_dim: int, max_steps: int = 256):
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.zeros((max_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_steps, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_steps, dtype=np.float32)
        self.dones = np.zeros(max_steps, dtype=np.float32)
        self.log_probs = np.zeros(max_steps, dtype=np.float32)
        self.values = np.zeros(max_steps, dtype=np.float32)

        # Filled by compute_gae
        self.advantages = np.zeros(max_steps, dtype=np.float32)
        self.returns = np.zeros(max_steps, dtype=np.float32)

    def add(self, state, action, reward, done, log_prob, value):
        if self.ptr >= self.max_steps:
            return  # buffer full — caller should check ready_to_train
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Generalized Advantage Estimation."""
        n = self.ptr
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, mini_batch_size: int):
        """Yield random mini-batch index arrays."""
        n = self.ptr
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            yield indices[start:end]

    def clear(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr


# ---------------------------------------------------------------------------
# Actor & Critic networks
# ---------------------------------------------------------------------------

class PPOActor(nn.Module):
    """Gaussian policy with tanh squashing for continuous actions."""

    def __init__(self, nb_states: int, nb_actions: int, hidden1: int = 256,
                 hidden2: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.mean_head = nn.Linear(hidden2, nb_actions)
        self.log_std_head = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-20.0, 2.0)
        return mean, log_std

    def get_action(self, state):
        """Sample action, return (tanh-squashed action, log_prob)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # pre-squash
        action = torch.tanh(u)

        # Log-prob with tanh correction
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        """Compute log_prob and entropy for given state-action pairs."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh to get pre-squash value
        u = torch.atanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PPOCritic(nn.Module):
    """State-value function V(s)."""

    def __init__(self, nb_states: int, hidden1: int = 256, hidden2: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# PPO algorithm
# ---------------------------------------------------------------------------

class PPO(AttackAlgorithm):
    """Proximal Policy Optimization for environment poisoning attacks.

    On-policy algorithm: collects rollout_size transitions, then runs
    ppo_epochs of mini-batch updates with clipped surrogate objective.
    """

    def __init__(self, nb_states: int, nb_actions: int, max_action: float,
                 hidden1: int = 256, hidden2: int = 256,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, ppo_epochs: int = 10,
                 mini_batch_size: int = 64, max_grad_norm: float = 0.5,
                 entropy_coef: float = 0.01, value_loss_coef: float = 0.5,
                 rollout_size: int = 256, **kwargs):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.max_action = max_action

        self.actor = PPOActor(nb_states, nb_actions, hidden1, hidden2).to(device)
        self.critic = PPOCritic(nb_states, hidden1, hidden2).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self._rollout_size = rollout_size

        self._rollout = RolloutBuffer(nb_states, nb_actions, max_steps=rollout_size)
        self._loss_log: List[list] = []

        # Stashed from last act() call for store_transition
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_obs = None

    # ------------------------------------------------------------------
    # AttackAlgorithm ABC
    # ------------------------------------------------------------------

    @property
    def is_on_policy(self) -> bool:
        return True

    @property
    def warmup_episodes(self) -> int:
        return 0  # PPO explores via entropy, no warmup needed

    def act(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_t)
            value = self.critic(state_t)

        action_np = action.cpu().numpy().squeeze(0)
        self._last_log_prob = log_prob.cpu().item()
        self._last_value = value.cpu().item()
        self._last_obs = state.copy()

        # Scale to max_action range
        return action_np * self.max_action

    def store_transition(self, obs, action, next_obs, reward, done):
        # Store action in [-1, 1] range (undo max_action scaling)
        action_normalized = np.array(action) / self.max_action
        action_normalized = np.clip(action_normalized, -1.0, 1.0)
        self._rollout.add(
            obs, action_normalized, reward, done,
            self._last_log_prob, self._last_value,
        )

    def ready_to_train(self) -> bool:
        return self._rollout.ptr >= self._rollout_size

    def update(self, episode: int = 0) -> Dict[str, float]:
        # Bootstrap last value for GAE
        if self._last_obs is not None:
            state_t = torch.FloatTensor(self._last_obs.reshape(1, -1)).to(device)
            with torch.no_grad():
                last_value = self.critic(state_t).cpu().item()
        else:
            last_value = 0.0

        self._rollout.compute_gae(last_value, self.gamma, self.gae_lambda)

        n = self._rollout.ptr
        # Convert to tensors
        states = torch.FloatTensor(self._rollout.states[:n]).to(device)
        actions = torch.FloatTensor(self._rollout.actions[:n]).to(device)
        old_log_probs = torch.FloatTensor(self._rollout.log_probs[:n]).to(device)
        advantages = torch.FloatTensor(self._rollout.advantages[:n]).to(device)
        returns = torch.FloatTensor(self._rollout.returns[:n]).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for batch_idx in self._rollout.get_batches(self.mini_batch_size):
                batch_idx = torch.LongTensor(batch_idx).to(device)
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Actor loss (clipped surrogate)
                new_lp, entropy = self.actor.evaluate(b_states, b_actions)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                # Critic loss
                values = self.critic(b_states).squeeze(-1)
                critic_loss = self.value_loss_coef * nn.functional.mse_loss(values, b_returns)

                # Update actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                # Update critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_updates += 1

        self._rollout.clear()

        avg_actor = total_actor_loss / max(num_updates, 1)
        avg_critic = total_critic_loss / max(num_updates, 1)
        self._loss_log.append([episode, 0, avg_critic, avg_actor])
        return {'actor_loss': avg_actor, 'critic_loss': avg_critic}

    def get_loss_log(self) -> list:
        return self._loss_log

    def save(self, path: str) -> None:
        torch.save(self.actor.state_dict(), path + "_ppo_actor")
        torch.save(self.critic.state_dict(), path + "_ppo_critic")
        torch.save(self.actor_optim.state_dict(), path + "_ppo_actor_optimizer")
        torch.save(self.critic_optim.state_dict(), path + "_ppo_critic_optimizer")

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(path + "_ppo_actor", map_location=torch.device('cpu')))
        self.critic.load_state_dict(
            torch.load(path + "_ppo_critic", map_location=torch.device('cpu')))
        self.actor_optim.load_state_dict(
            torch.load(path + "_ppo_actor_optimizer", map_location=torch.device('cpu')))
        self.critic_optim.load_state_dict(
            torch.load(path + "_ppo_critic_optimizer", map_location=torch.device('cpu')))
