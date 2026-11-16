"""DQN victim for continuous-state environments (e.g. LunarLander).

Ported from DBB-EPA/src_lunarlander/victim/victim_dqn.py to the VictimAlgorithm
interface. Same network (8 -> 150 -> 120 -> actions), RMSprop, Huber loss, gradient
clamp, replay buffer, and target-network refresh every 10 episodes.

Two deliberate differences from the original, both noted where they occur:
  * the bootstrap target is masked on true termination (the original never masked);
  * the replay buffer is a pre-allocated array instead of a list of tensors (same
    uniform sampling, much faster).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.victim_algorithm import VictimAlgorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, fc1: int = 150, fc2: int = 120):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class VictimDQN(VictimAlgorithm):
    """Deep Q-learning victim. States are float vectors; actions are discrete."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        batch_size: int = 256,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.996,
        target_update: int = 10,
        learning_rate: float = 1e-3,
        buffer_size: int = 200_000,
        max_steps: Optional[int] = None,
        **_unused: Any,
    ):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self._trajectory_buffer: List[tuple] = []
        self._init_structures()

    @classmethod
    def from_env(cls, env: Any, **kwargs: Any) -> "VictimDQN":
        """Build from an environment with a Box observation space and Discrete actions."""
        return cls(
            int(np.prod(env.observation_space.shape)), int(env.action_space.n), **kwargs
        )

    def _init_structures(self) -> None:
        self.policy_net = _QNet(self.state_dim, self.n_actions).to(device)
        self.target_net = _QNet(self.state_dim, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(
            self.policy_net.parameters(), lr=self.learning_rate
        )
        n = self.buffer_size
        self._s = np.zeros((n, self.state_dim), dtype=np.float32)
        self._s2 = np.zeros((n, self.state_dim), dtype=np.float32)
        self._a = np.zeros(n, dtype=np.int64)
        self._r = np.zeros(n, dtype=np.float32)
        self._nd = np.zeros(
            n, dtype=np.float32
        )  # 1 if the episode did not terminate here
        self._ptr = self._size = 0
        self.steps_done = 0
        self.episodes_done = 0

    # --- acting ---

    def _epsilon(self) -> float:
        # As written in the original: an exponential in *steps* with constant eps_decay.
        # With eps_decay = 0.996 this reaches eps_end after a handful of steps.
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay
        )

    def greedy_actions_for(self, states: np.ndarray) -> np.ndarray:
        """Greedy (argmax-Q) action for each row of states. Evaluation helper."""
        with torch.no_grad():
            q = self.policy_net(
                torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
            )
        return q.argmax(dim=1).cpu().numpy()

    def act(self, state: Any) -> int:
        eps = self._epsilon()
        self.steps_done += 1
        if np.random.random() > eps:
            return int(
                self.greedy_actions_for(
                    np.asarray(state, dtype=np.float32).reshape(1, -1)
                )[0]
            )
        return int(np.random.randint(self.n_actions))

    # --- learning ---

    def _optimize(self) -> None:
        if self._size < self.batch_size:
            return
        idx = np.random.randint(0, self._size, size=self.batch_size)
        s = torch.as_tensor(self._s[idx], device=device)
        a = torch.as_tensor(self._a[idx], device=device).unsqueeze(1)
        r = torch.as_tensor(self._r[idx], device=device)
        s2 = torch.as_tensor(self._s2[idx], device=device)
        nd = torch.as_tensor(self._nd[idx], device=device)

        q = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            # Difference from the original: no bootstrapping past a terminal state.
            target = r + self.gamma * nd * self.target_net(s2).max(1)[0]
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.policy_net.parameters():
            p.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _run_episode(self, env: Any) -> Dict[str, Any]:
        obs = env.reset()
        state = np.asarray(
            obs[0] if isinstance(obs, tuple) else obs, dtype=np.float32
        ).reshape(-1)
        limit = self.max_steps or getattr(env, "max_steps", None) or 1000
        score, prev_state, prev_action = 0.0, None, None
        t = 0
        for t in range(int(limit)):
            action = self.act(state)
            result = env.step(action)
            next_state, reward, terminated, truncated, _ = (
                result if len(result) == 5 else (*result[:3], False, result[3])
            )
            next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)

            i = self._ptr
            self._s[i], self._a[i], self._r[i], self._s2[i] = (
                state,
                action,
                reward,
                next_state,
            )
            self._nd[i] = 0.0 if terminated else 1.0
            self._ptr = (self._ptr + 1) % self.buffer_size
            self._size = min(self._size + 1, self.buffer_size)
            self._optimize()

            if prev_state is not None:
                self._trajectory_buffer.append((prev_state, prev_action, state, action))
            prev_state, prev_action = state, action
            state = next_state
            score += float(reward)
            if terminated or truncated:
                break

        if self.episodes_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.episodes_done += 1
        return {"reward": score, "length": t + 1}

    def train(self, env: Any, num_episodes: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"episode_rewards": [], "episode_lengths": []}
        for _ in range(num_episodes):
            ep = self._run_episode(env)
            stats["episode_rewards"].append(ep["reward"])
            stats["episode_lengths"].append(ep["length"])
        self.last_stats = stats
        return stats

    def reset(self) -> None:
        self._init_structures()
        self._trajectory_buffer = []

    # --- what the attacker may observe ---

    def get_trajectories(self) -> list:
        """(prev_state, prev_action, state, action) tuples since the last call."""
        out = self._trajectory_buffer
        self._trajectory_buffer = []
        return out

    def get_behavior_trace(self):
        """Not defined for continuous states (there is no per-state table)."""
        return None

    def get_policy_matrix(self):
        """Not defined for continuous states."""
        return None

    # --- persistence ---

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), f"{path}_dqn.pt")

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(
            torch.load(f"{path}_dqn.pt", map_location=device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
