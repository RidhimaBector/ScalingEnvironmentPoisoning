"""Measures of how close a victim is to the target behaviour.

Functions that take a policy table are ground truth (evaluation, or the white-box
setting). Functions that take a behaviour trace or trajectories use only what a
black-box attacker can observe.
"""
import numpy as np
import ot
from scipy.special import softmax



def target_action_accuracy(target: np.ndarray, policy_matrix: np.ndarray) -> float:
    """Share of target states whose greedy action (argmax of the policy table row) is
    the target action. Ground truth: needs the victim's policy table."""
    return float(policy_accuracy(target, policy_matrix)[1])


def policy_accuracy(target: np.ndarray, policy_matrix: np.ndarray):
    """Compare a victim's policy table with the target policy.

    Returns (done, accuracy, accuracy_softmax, accuracy_softmax_complete):
      accuracy                    share of target states with the target's greedy action
      accuracy_softmax_complete   mean softmax probability of the target action
      accuracy_softmax            the same, over the states that already match
      done                        1 if accuracy is 1.0
    """
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
    accuracy_softmax = (float(np.sum(accuracy_softmax_elementwise) / sum_accuracy_elementwise)
                        if sum_accuracy_elementwise != 0 else 0.0)

    done = 1 if accuracy == 1.0 else 0

    return done, accuracy, accuracy_softmax, accuracy_softmax_complete


def accuracy_from_behavior_trace(behavior_trace: np.ndarray, target: np.ndarray) -> float:
    """Blackbox-safe accuracy: uses last observed greedy action per state."""
    target_map = np.sum(target, axis=1)
    target_actions = np.argmax(target, axis=1)
    trace_actions = behavior_trace[:, 1].astype(int)
    accuracy = np.sum(target_map * (trace_actions == target_actions)) / np.sum(target_map)
    return float(accuracy)


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


def accuracy_from_trajectories(trajectories, target: np.ndarray) -> float:
    """Blackbox-safe accuracy from observed (pre_s, pre_a, s, a) tuples.

    For every target state: the fraction of the victim's observed actions in
    that state that equal the target action (0 if the state was never visited).
    Returns the mean over target states.
    """
    nS, nA = target.shape
    counts = np.zeros((nS, nA))
    for pre_s, pre_a, s, a in trajectories:
        counts[int(pre_s), int(pre_a)] += 1
    if trajectories:
        counts[int(trajectories[-1][2]), int(trajectories[-1][3])] += 1
    target_map = np.sum(target, axis=1)
    target_actions = np.argmax(target, axis=1)
    visits = counts.sum(axis=1)
    hit = counts[np.arange(nS), target_actions]
    frac = np.where(visits > 0, hit / np.maximum(visits, 1), 0.0)
    return float(np.sum(target_map * frac) / np.sum(target_map))


def greedy_path(T: np.ndarray, greedy_actions: np.ndarray, start: int, goal: int, max_len: int = 64) -> list:
    """Most-likely state sequence when following greedy_actions under dynamics T (nA, nS, nS)."""
    s = int(start)
    path = [s]
    for _ in range(max_len):
        s = int(np.argmax(T[int(greedy_actions[s])][s]))
        path.append(s)
        if s == goal:
            break
    return path
