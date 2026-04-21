import numpy as np
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
