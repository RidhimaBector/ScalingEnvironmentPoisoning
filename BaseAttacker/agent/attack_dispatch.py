"""Attack dispatch: applies attack actions to victim environments.

Simple component that clips and applies the attack agent's action
(perturbation vector) to all victim environments in the population.
"""

import numpy as np

from agent.victim_system import VictimSystem


class AttackDispatch:
    """Applies attack perturbation actions to the victim population.

    The default implementation clips the action to [-1, 1] and delegates
    to VictimSystem.apply_perturbation(). Subclass to implement custom
    dispatch logic (e.g., different perturbation per victim).
    """

    def apply(self, population: VictimSystem, action: np.ndarray) -> None:
        """Apply the attack action to all victim environments.

        Args:
            population: VictimSystem managing K victim agents.
            action: Perturbation vector from the attack agent, typically in [-1, 1].
        """
        clipped_action = np.clip(action, -1.0, 1.0)
        population.apply_perturbation(clipped_action)
