"""Policy-gradient victims for continuous-state environments (e.g. LunarLander).

Two learners behind the VictimAlgorithm interface, standing in for the VPG and A2C
victims of DBB-EPA/src_lunarlander:

  VictimVPG  REINFORCE with a neural policy; one update per episode on normalised
             discounted returns.
  VictimA2C  the same policy plus a value network; the update uses the advantage
             (return minus predicted value) and also fits the value network.

Both are written fresh against the interface (they are not line-by-line ports).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.victim_algorithm import VictimAlgorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp(inp: int, out: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class VictimVPG(VictimAlgorithm):
    """REINFORCE with a neural softmax policy."""

    use_critic = False

    def __init__(self, state_dim: int, n_actions: int, gamma: float = 0.99,
                 learning_rate: float = 1e-3, max_steps: Optional[int] = None, **_unused: Any):
        self.state_dim, self.n_actions = int(state_dim), int(n_actions)
        self.gamma, self.learning_rate, self.max_steps = gamma, learning_rate, max_steps
        self._trajectory_buffer: List[tuple] = []
        self._init_structures()

    @classmethod
    def from_env(cls, env: Any, **kwargs: Any):
        return cls(int(np.prod(env.observation_space.shape)), int(env.action_space.n), **kwargs)

    def _init_structures(self) -> None:
        self.policy = _mlp(self.state_dim, self.n_actions).to(device)
        params = list(self.policy.parameters())
        if self.use_critic:
            self.value = _mlp(self.state_dim, 1).to(device)
            params += list(self.value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

    # --- acting ---

    def _logits(self, states: np.ndarray) -> torch.Tensor:
        return self.policy(torch.as_tensor(np.asarray(states, dtype=np.float32), device=device))

    def act(self, state: Any) -> int:
        with torch.no_grad():
            probs = F.softmax(self._logits(np.asarray(state).reshape(1, -1)), dim=1)[0].cpu().numpy()
        return int(np.random.choice(self.n_actions, p=probs / probs.sum()))

    def greedy_actions_for(self, states: np.ndarray) -> np.ndarray:
        """Most likely action for each row of states. Evaluation helper."""
        with torch.no_grad():
            return self._logits(states).argmax(dim=1).cpu().numpy()

    # --- learning ---

    def _update(self, states: List[np.ndarray], actions: List[int], rewards: List[float]) -> None:
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns_t = torch.as_tensor(returns[::-1], dtype=torch.float32, device=device)
        s = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=device)
        logp = F.log_softmax(self.policy(s), dim=1).gather(1, a.unsqueeze(1)).squeeze(1)
        if self.use_critic:
            v = self.value(s).squeeze(1)
            scale = returns_t.std() + 1e-6
            adv = ((returns_t - v.detach()) / scale)
            loss = -(logp * adv).mean() + 0.5 * F.mse_loss(v / scale, returns_t / scale)
        else:
            adv = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)
            loss = -(logp * adv).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

    def _run_episode(self, env: Any) -> Dict[str, Any]:
        obs = env.reset()
        state = np.asarray(obs[0] if isinstance(obs, tuple) else obs, dtype=np.float32).reshape(-1)
        limit = self.max_steps or getattr(env, 'max_steps', None) or 1000
        states, actions, rewards = [], [], []
        prev_state, prev_action = None, None
        for _ in range(int(limit)):
            action = self.act(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = result if len(result) == 5 else (*result[:3], False, result[3])
            states.append(state); actions.append(action); rewards.append(float(reward))
            if prev_state is not None:
                self._trajectory_buffer.append((prev_state, prev_action, state, action))
            prev_state, prev_action = state, action
            state = np.asarray(next_state, dtype=np.float32).reshape(-1)
            if terminated or truncated:
                break
        self._update(states, actions, rewards)
        return {'reward': float(np.sum(rewards)), 'length': len(rewards)}

    def train(self, env: Any, num_episodes: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {'episode_rewards': [], 'episode_lengths': []}
        for _ in range(num_episodes):
            ep = self._run_episode(env)
            stats['episode_rewards'].append(ep['reward'])
            stats['episode_lengths'].append(ep['length'])
        self.last_stats = stats
        return stats

    def reset(self) -> None:
        self._init_structures()
        self._trajectory_buffer = []

    # --- what the attacker may observe ---

    def get_trajectories(self) -> list:
        out = self._trajectory_buffer
        self._trajectory_buffer = []
        return out

    def get_behavior_trace(self):
        return None   # no per-state table for continuous states

    def get_policy_matrix(self):
        return None

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), f"{path}_policy.pt")

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(f"{path}_policy.pt", map_location=device))


class VictimA2C(VictimVPG):
    """Advantage actor-critic: VictimVPG with a learned value baseline."""

    use_critic = True
