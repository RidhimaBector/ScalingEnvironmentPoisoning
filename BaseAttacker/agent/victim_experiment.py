"""Victim experiment tracker for Sacred logging.

Logs per-victim metrics (accuracy, convergence, behavior trace stats) to
a dedicated Sacred experiment instance, separate from the attack tracker.

Injected into VictimSystem and called automatically during run_experiments().
Works without Sacred/MongoDB (gracefully degrades to no-op).
"""

from typing import Any, Dict, List, Optional

import numpy as np


class VictimExperimentTracker:
    """Tracks per-victim metrics in a dedicated Sacred experiment.

    If no Sacred run is provided, all methods are no-ops (logging disabled).

    Args:
        sacred_run: Sacred _run object for the victim experiment, or None.
    """

    def __init__(self, sacred_run: Any = None):
        self._run = sacred_run
        self._step = 0
        self._per_victim_steps: Dict[int, int] = {}

    @property
    def enabled(self) -> bool:
        """Whether Sacred logging is active."""
        return self._run is not None

    def log_victim_step(
        self, victim_idx: int, metrics: Dict[str, float]
    ) -> None:
        """Log metrics for a single victim at the current step.

        Called by VictimSystem.run_experiments() after each victim trains.

        Args:
            victim_idx: Index of the victim (0 to K-1).
            metrics: Dict with accuracy, accuracy_sftmx, etc.
        """
        if not self.enabled:
            return

        step = self._per_victim_steps.get(victim_idx, 0)

        for name, value in metrics.items():
            self._run.log_scalar(f"victim_{victim_idx}.{name}", value, step)

        self._per_victim_steps[victim_idx] = step + 1

    def log_population_step(self, metrics: Dict[str, float]) -> None:
        """Log population-level metrics (mean/min/max accuracy).

        Args:
            metrics: Dict with aggregated accuracy metrics.
        """
        if not self.enabled:
            return

        for name, value in metrics.items():
            if not isinstance(value, list):
                self._run.log_scalar(f"population.{name}", value, self._step)

        self._step += 1

    def log_info(self, key: str, value: Any) -> None:
        """Log arbitrary info to the Sacred run.

        Args:
            key: Info key.
            value: Info value.
        """
        if self.enabled:
            self._run.info[key] = value


class AttackExperimentTracker:
    """Tracks attack-level metrics in a dedicated Sacred experiment.

    Logs DDPG loss, mean accuracy, effort, best/worst model stats,
    and checkpoint paths. Works without Sacred (no-op if run is None).

    Args:
        sacred_run: Sacred _run object for the attack experiment, or None.
    """

    def __init__(self, sacred_run: Any = None):
        self._run = sacred_run
        self._episode_step = 0

    @property
    def enabled(self) -> bool:
        """Whether Sacred logging is active."""
        return self._run is not None

    def log_timestep(
        self,
        episode: int,
        timestep: int,
        info: Dict[str, Any],
    ) -> None:
        """Log metrics for a single attack timestep.

        Args:
            episode: Current episode number.
            timestep: Current timestep within episode.
            info: Info dict from AttackEnvironment.step().
        """
        if not self.enabled:
            return

        global_step = episode * 1000 + timestep
        for name, value in info.items():
            if isinstance(value, (int, float)):
                self._run.log_scalar(f"timestep.{name}", value, global_step)

    def log_episode(
        self,
        episode: int,
        metrics: Optional[Dict[str, float]] = None,
        ddpg_loss: Optional[float] = None,
    ) -> None:
        """Log episode-level metrics.

        Args:
            episode: Episode number.
            metrics: Optional dict of episode metrics.
            ddpg_loss: Optional DDPG loss value.
        """
        if not self.enabled:
            return

        if metrics:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._run.log_scalar(f"episode.{name}", value, episode)

        if ddpg_loss is not None:
            self._run.log_scalar("ddpg_loss", ddpg_loss, episode)

        self._episode_step = episode

    def log_checkpoint(self, episode: int, accuracy: float, path: str) -> None:
        """Log checkpoint information.

        Args:
            episode: Episode number.
            accuracy: Current accuracy at checkpoint.
            path: Path where checkpoint was saved.
        """
        if self.enabled:
            self._run.info[f"checkpoint_{episode}"] = {
                'episode': episode,
                'accuracy': accuracy,
                'path': path,
            }

    def log_info(self, key: str, value: Any) -> None:
        """Log arbitrary info to the Sacred run.

        Args:
            key: Info key.
            value: Info value.
        """
        if self.enabled:
            self._run.info[key] = value
