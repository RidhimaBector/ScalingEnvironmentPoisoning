"""Push exploration for the attacker.

Per-step random actions average out to almost no change in the environment, so the
attacker never sees what a strongly modified environment does to the victim. In a
push episode every perturbation dimension is instead driven in one fixed direction
(down / hold / up) for the whole episode.

With `adaptive=True` the directions are drawn from per-dimension probabilities that
are updated by the cross-entropy method: after every `batch` push episodes they move
towards the `elite` episodes with the highest reward. This turns exploration into a
search over "which way to push each dimension", guided only by the attacker's own
reward.

Works for any Box perturbation space; nothing here knows what a dimension means.
"""

from typing import Optional

import numpy as np


class PushExploration:
    def __init__(
        self,
        action_shape,
        prob: float = 0.2,
        adaptive: bool = True,
        batch: int = 16,
        elite: int = 4,
        lr: float = 0.5,
        floor: float = -1.0,
        noise: float = 0.2,
    ):
        self.shape = tuple(action_shape)
        self.prob = prob
        self.adaptive = adaptive
        self.batch, self.elite, self.lr, self.noise = batch, elite, lr, noise
        n_dims = int(np.prod(self.shape))
        # A fixed floor of 0.05 means each dimension leaves its best direction 10% of
        # the time, so with 36 dimensions only ~2% of pushes try the best combination.
        # floor < 0 scales it with the number of dimensions.
        self.floor = floor if floor >= 0 else 0.25 / n_dims
        self.probs = np.full(
            (n_dims, 3), 1.0 / 3.0
        )  # P(down), P(hold), P(up) per dimension
        self._pending = []  # (episode reward, direction indices)
        self._current: Optional[np.ndarray] = None

    def start_episode(self, warmup: bool) -> bool:
        """Decide whether the coming episode is a push episode and pick its directions."""
        is_push = self.prob > 0 and np.random.rand() < (0.5 if warmup else self.prob)
        if self.adaptive:
            idx = np.array([np.random.choice(3, p=p) for p in self.probs])
        else:
            idx = np.random.choice(3, size=len(self.probs))
        self._current = idx if is_push else None
        return is_push

    def action(self) -> np.ndarray:
        """The push action for this step: the episode's directions plus a little noise."""
        direction = self._current.astype(float) - 1.0
        return np.clip(
            direction + np.random.normal(0.0, self.noise, size=direction.shape),
            -1.0,
            1.0,
        ).reshape(self.shape)

    def end_episode(self, episode_reward: float) -> None:
        """Feed back the reward of a push episode; update the probabilities when due."""
        if self._current is None or not self.adaptive:
            return
        self._pending.append((episode_reward, self._current))
        if len(self._pending) >= self.batch:
            best = sorted(self._pending, key=lambda item: -item[0])[: self.elite]
            freq = np.mean([np.eye(3)[idx] for _, idx in best], axis=0)
            self.probs = np.maximum(
                (1 - self.lr) * self.probs + self.lr * freq, self.floor
            )
            self.probs /= self.probs.sum(axis=1, keepdims=True)
            self._pending = []
