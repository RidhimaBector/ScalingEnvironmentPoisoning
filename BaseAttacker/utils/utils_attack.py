import math
import numpy as np
import ot
from scipy.special import softmax

from .utils_op import *


def Attack_Done_Identify(target: np.ndarray, policy_matrix: np.ndarray, policy: np.ndarray = None):
    target_map = np.sum(target, axis=1)
    total = np.sum(target_map)
    index_target = np.argmax(target, axis=1)
    accuracy_elementwise = target_map*(index_target == np.argmax(policy_matrix, axis=1))
    sum_accuracy_elementwise = np.sum(accuracy_elementwise)
    accuracy = sum_accuracy_elementwise/total

    pm_softmax = softmax(policy_matrix, axis=1)
    accuracy_softmax_complete_elementwise = target_map*( pm_softmax[np.arange(policy_matrix.shape[0]),index_target] )
    accuracy_softmax_complete = np.sum( accuracy_softmax_complete_elementwise )/total

    accuracy_softmax_elementwise = accuracy_elementwise * accuracy_softmax_complete_elementwise
    accuracy_softmax = np.where( (sum_accuracy_elementwise!=0), np.sum(accuracy_softmax_elementwise)/sum_accuracy_elementwise, 0.0).item()

    done = 1 if accuracy == 1.0 else 0

    return done, accuracy, accuracy_softmax, accuracy_softmax_complete


def Attack_Effort(current_env_dynamics, env):
    prev_env_dynamics = current_env_dynamics.copy()
    current_env_dynamics_updated = env.env_dynamics.copy()
    action_clipped = current_env_dynamics_updated - prev_env_dynamics
    effort = np.mean(np.abs(action_clipped))

    return effort, current_env_dynamics_updated


def Attack_Cost_Compute_W(env, init_T, policy_matrix, target, cost_matrix, distance_type=0):
    """Wasserstein distance rate between the victim's Markov chain and the target's.

    Generalized over env.nS and env.nA — no hardcoded grid sizes.

    Args:
        env: VictimEnvironment with nS, nA, T attributes.
        init_T: Initial (unperturbed) transition tensor of shape (nA, nS, nS).
        policy_matrix: Victim Q-table or policy of shape (nS, nA).
        target: Target policy matrix of shape (nS, nA).
        cost_matrix: Ground-metric matrix of shape (nS*nA, nS*nA).
        distance_type: 0=complete, 1=grid, 2=behavior.

    Returns:
        Scalar Wasserstein distance cost.
    """
    policy = softmax(policy_matrix, axis=1)
    sxa = env.nS * env.nA

    target_map = np.repeat(np.sum(target, axis=1).reshape(1, env.nS), env.nA, axis=1)
    target_policy = (target + 0.001) / (1 + 0.001 * env.nA)
    T = env.T.copy()

    init_T_matrix = np.transpose(init_T, (1, 0, 2)).reshape(sxa, env.nS)
    init_T_matrix = np.repeat(init_T_matrix, env.nA, axis=1)
    policy_matrix_rep = np.repeat(policy.reshape(1, sxa), sxa, axis=0)
    target_policy_matrix = np.repeat(target_policy.reshape(1, sxa), sxa, axis=0)
    target_map_rep = np.tile(target_map, (sxa, 1))
    P_star = target_map_rep * (init_T_matrix * target_policy_matrix) + (1 - target_map_rep) * (init_T_matrix * policy_matrix_rep)

    T_matrix = np.transpose(T, (1, 0, 2)).reshape(sxa, env.nS)
    T_matrix = np.repeat(T_matrix, env.nA, axis=1)
    if distance_type == 0:
        P = T_matrix * policy_matrix_rep
    elif distance_type == 1:
        P = target_map_rep * (T_matrix * target_policy_matrix) + (1 - target_map_rep) * (T_matrix * policy_matrix_rep)
    else:
        P = init_T_matrix * policy_matrix_rep

    # Initial distribution: uniform over first env.nA state-action pairs (first row of grid)
    initial_distribution = np.zeros((1, sxa))
    initial_distribution[0, :env.nA] = 1.0 / env.nA

    # N-th step probabilities (10 matrix multiplications to approximate stationary behaviour)
    nth_step_prob_star = np.matmul(P_star, P_star)
    nth_step_prob = np.matmul(P, P)
    for _ in range(8):
        nth_step_prob_star = np.matmul(nth_step_prob_star, P_star)
        nth_step_prob = np.matmul(nth_step_prob, P)
    nth_step_path_star = np.around(np.squeeze(np.matmul(initial_distribution, nth_step_prob_star)), 10)
    nth_step_path = np.around(np.squeeze(np.matmul(initial_distribution, nth_step_prob)), 10)

    # Normalize
    nth_step_path_star = nth_step_path_star / np.sum(nth_step_path_star)
    nth_step_path = nth_step_path / np.sum(nth_step_path)

    transport_matrix = ot.emd(nth_step_path_star, nth_step_path, M=cost_matrix)
    return float(np.sum(cost_matrix * transport_matrix))


def Attack_Cost_Compute_K(env, init_T, policy_matrix, target, cost_matrix, distance_type=0):
    """KL-rate distance between the victim's Markov chain and the target's.

    Generalized over env.nS and env.nA — no hardcoded grid sizes.

    Args:
        env: VictimEnvironment with nS, nA, T attributes.
        init_T: Initial (unperturbed) transition tensor of shape (nA, nS, nS).
        policy_matrix: Victim Q-table or policy of shape (nS, nA).
        target: Target policy matrix of shape (nS, nA).
        cost_matrix: Unused; kept for API symmetry with Attack_Cost_Compute_W.
        distance_type: 0=complete, 1=grid, 2=behavior.

    Returns:
        Scalar KL-rate distance cost.
    """
    policy = softmax(policy_matrix, axis=1)
    sxa = env.nS * env.nA

    target_map = np.repeat(np.sum(target, axis=1).reshape(1, env.nS), env.nA, axis=1)
    target_policy = (target + 0.001) / (1 + 0.001 * env.nA)
    T = env.T.copy()

    init_T_matrix = np.transpose(init_T, (1, 0, 2)).reshape(sxa, env.nS)
    init_T_matrix = np.repeat(init_T_matrix, env.nA, axis=1)
    policy_matrix_rep = np.repeat(policy.reshape(1, sxa), sxa, axis=0)
    target_policy_matrix = np.repeat(target_policy.reshape(1, sxa), sxa, axis=0)
    target_map_rep = np.tile(target_map, (sxa, 1))
    P_star = target_map_rep * (init_T_matrix * target_policy_matrix) + (1 - target_map_rep) * (init_T_matrix * policy_matrix_rep)

    T_matrix = np.transpose(T, (1, 0, 2)).reshape(sxa, env.nS)
    T_matrix = np.repeat(T_matrix, env.nA, axis=1)
    if distance_type == 0:
        P = T_matrix * policy_matrix_rep
    elif distance_type == 1:
        P = target_map_rep * (T_matrix * target_policy_matrix) + (1 - target_map_rep) * (T_matrix * policy_matrix_rep)
    else:
        P = init_T_matrix * policy_matrix_rep

    # KL divergence per (state, action) pair
    DKL_matrix = np.sum(P * np.log(np.where(P > 0, P / np.maximum(P_star, 1e-300), 1.0)), axis=1).reshape(env.nS, env.nA)

    # Stationary distribution via eigenvector of P^T
    w, v = np.linalg.eig(P.T)
    j_stationary = np.argmin(abs(w - 1.0))
    q_stationary = v[:, j_stationary].real
    q_stationary /= q_stationary.sum()
    cost = float(np.sum(q_stationary.reshape(env.nS, env.nA) * DKL_matrix))

    return cost
