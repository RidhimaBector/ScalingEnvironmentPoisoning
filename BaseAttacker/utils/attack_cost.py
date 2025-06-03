from typing import Dict, Optional, Union

import numpy as np
import ot


class AttackCost:
    """
    A service class for computing various distance metrics between policies
    regardless of environment characteristics.
    """

    def __init__(self):
        pass

    def kullback_leibler_divergence(
        self,
        current_policy: Union[np.ndarray, Dict],
        target_policy: Union[np.ndarray, Dict],
        transition_model: Optional[np.ndarray] = None,
        policy_approximator: Optional[callable] = None,
        **kwargs
    ) -> float:
        """
        Compute KL divergence between current and target policies.

        Args:
            current_policy: Current policy (matrix or dict)
            target_policy: Target policy (matrix or dict)
            transition_model: Optional transition dynamics
            policy_approximator: Optional function to approximate policy
            **kwargs: Additional arguments for specific implementations
        """
        if transition_model is not None:
            return self._kl_with_transitions(current_policy, target_policy, transition_model, **kwargs)
        elif policy_approximator is not None:
            return self._kl_with_approximator(current_policy, target_policy, policy_approximator, **kwargs)
        else:
            return self._kl_direct(current_policy, target_policy, **kwargs)

    def wasserstein_distance(
        self,
        current_policy: Union[np.ndarray, Dict],
        target_policy: Union[np.ndarray, Dict],
        cost_matrix: Optional[np.ndarray] = None,
        transition_model: Optional[np.ndarray] = None,
        policy_approximator: Optional[callable] = None,
        **kwargs
    ) -> float:
        """
        Compute Wasserstein distance between current and target policies.

        Args:
            current_policy: Current policy (matrix or dict)
            target_policy: Target policy (matrix or dict)
            cost_matrix: Cost matrix for Wasserstein computation
            transition_model: Optional transition dynamics
            policy_approximator: Optional function to approximate policy
            **kwargs: Additional arguments for specific implementations
        """
        if transition_model is not None:
            return self._wasserstein_with_transitions(
                current_policy, target_policy, cost_matrix, transition_model, **kwargs
            )
        elif policy_approximator is not None:
            return self._wasserstein_with_approximator(
                current_policy, target_policy, cost_matrix, policy_approximator, **kwargs
            )
        else:
            return self._wasserstein_direct(current_policy, target_policy, cost_matrix, **kwargs)

    def _kl_direct(
        self,
        current_policy: Union[np.ndarray, Dict],
        target_policy: Union[np.ndarray, Dict],
        **kwargs
    ) -> float:
        """Direct KL computation between policies without transition model"""
        # Convert dict policies to arrays if needed
        if isinstance(current_policy, dict):
            current_policy = self._dict_to_array(current_policy)
        if isinstance(target_policy, dict):
            target_policy = self._dict_to_array(target_policy)

        return np.sum(current_policy * np.log(current_policy / target_policy))

    def _kl_with_transitions(
        self,
        current_policy: np.ndarray,
        target_policy: np.ndarray,
        transition_model: np.ndarray,
        **kwargs
    ) -> float:
        """KL computation using transition dynamics"""
        P = self._compute_transition_probability(current_policy, transition_model)
        P_star = self._compute_transition_probability(target_policy, transition_model)

        return np.sum(P * np.log(P / P_star))

    def _kl_with_approximator(
        self,
        current_policy: Union[np.ndarray, Dict],
        target_policy: Union[np.ndarray, Dict],
        policy_approximator: callable,
        **kwargs
    ) -> float:
        """KL computation using policy approximation"""
        current_approx = policy_approximator(current_policy)
        target_approx = policy_approximator(target_policy)

        return self._kl_direct(current_approx, target_approx)

    def _wasserstein_direct(
        self,
        current_policy: Union[np.ndarray, Dict],
        target_policy: Union[np.ndarray, Dict],
        cost_matrix: np.ndarray,
        **kwargs
    ) -> float:
        """Direct Wasserstein computation between policies"""
        if isinstance(current_policy, dict):
            current_policy = self._dict_to_array(current_policy)
        if isinstance(target_policy, dict):
            target_policy = self._dict_to_array(target_policy)

        return ot.emd2(current_policy, target_policy, cost_matrix)

    def _wasserstein_with_transitions(
        self,
        current_policy: np.ndarray,
        target_policy: np.ndarray,
        cost_matrix: np.ndarray,
        transition_model: np.ndarray,
        **kwargs
    ) -> float:
        """Wasserstein computation using transition dynamics"""
        P = self._compute_transition_probability(current_policy, transition_model)
        P_star = self._compute_transition_probability(target_policy, transition_model)

        return ot.emd2(P.flatten(), P_star.flatten(), cost_matrix)

    def _wasserstein_with_approximator(
        self,
        current_policy: Union[np.ndarray, Dict],
        target_policy: Union[np.ndarray, Dict],
        cost_matrix: np.ndarray,
        policy_approximator: callable,
        **kwargs
    ) -> float:
        """Wasserstein computation using policy approximation"""
        current_approx = policy_approximator(current_policy)
        target_approx = policy_approximator(target_policy)

        return self._wasserstein_direct(current_approx, target_approx, cost_matrix)

    @staticmethod
    def _dict_to_array(policy_dict: Dict) -> np.ndarray:
        """Convert a dictionary policy to numpy array format"""
        # Implementation depends on specific dictionary format
        raise NotImplementedError("Dictionary conversion must be implemented")

    @staticmethod
    def _compute_transition_probability(
        policy: np.ndarray,
        transition_model: np.ndarray
    ) -> np.ndarray:
        """Compute transition probabilities given policy and dynamics"""
        return np.multiply(transition_model, policy)

