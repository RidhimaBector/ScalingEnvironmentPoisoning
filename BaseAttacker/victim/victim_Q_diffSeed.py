import math
import random
import numpy as np
import copy
from scipy.special import softmax
import time

from collections import namedtuple
from collections import defaultdict
from itertools import count
import itertools

import os
from os.path import dirname, abspath

from victim.victim_Q import VictimQLearning

import sys


from utils import utils_buf, utils_op, utils_attack
from envs.target_def import TARGET

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

if "../" not in sys.path:
    sys.path.append("../")

#LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
T_max = config.VICTIM.TMAX


class QLearning_DiffSeed(VictimQLearning):


    def __init__(self, env, MEMORY_SIZE, discount_factor=1.0, alpha=0.1, epsilon=0.1):
        super().__init__(env, MEMORY_SIZE, discount_factor, alpha, epsilon)
        seed = int( time.time() )
        self.env.seed(seed)
        np.random.seed(seed)
        self.Q = np.zeros((16,self.env.action_space.n)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buf.Memory(MEMORY_SIZE)


    def train_for_eval(self, num_episodes: int):

        # The policy we're following
        #policy = self.MakeEpsilonGreedyPolicy()
        Q_matrix = self.Q #utils_op.DicQ_To_MatrixQ(self.Q, self.env)
        victim_transitions = np.ones((16,2)) * 4
        victim_transitions[:,0] = np.arange(16)

        #stats_timestep = []
        accuracy_episode = []
        accuracy_softmax_episode = []
        accuracy_softmax_complete_episode = []

        for i_episode in range(num_episodes):
            # logger
            trajectory_list = []
            score = 0

            # Reset env
            state = self.env.reset()

            for t in itertools.count(): #for t in range(10):
                # logger
                t_sample = []

                # Take a step
                action_probs = softmax(Q_matrix[state]) #action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                victim_transitions[state,1] = action

                # TD Update
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

                # save transitions
                """if t!=0:
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)"""

                state = copy.deepcopy(next_state)

                if done:
                    done_attack, accuracy, accuracy_softmax, accuracy_softmax_complete = utils_attack.Attack_Done_Identify(self.env, TARGET, self.Q)
                    accuracy_episode.append(accuracy)
                    accuracy_softmax_episode.append(accuracy_softmax)
                    accuracy_softmax_complete_episode.append(accuracy_softmax_complete)
                    break

            # Trajectory to MEMORY with head_padding
            """if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = 0
                padding_action = 0
                n_padding = LEN_TRAJECTORY - len(trajectory_list)
                for i in range(n_padding):
                    self.MEM.push(padding_state, padding_action)
                for i in range(len(trajectory_list)):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)
            else:
                for i in range(LEN_TRAJECTORY):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)"""

        return accuracy_episode, accuracy_softmax_episode, accuracy_softmax_complete_episode, victim_transitions #[[0,1,2,3,7,11], :]


if __name__ == "__main__":
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()

    victim_args = {
        "env": env,
        "MEMORY_SIZE": 100,
        "discount_factor": 1.0,
        "alpha": 0.1,
        "epsilon": 0.1,
    }

    victim = QLearning_DiffSeed(**victim_args)
    victim.Train_Model(10)
    victim.Show_PolicyQ()
    victim.Eval_Model()

    print(f"Size of Memory = {victim.MEM.__len__()}")
    print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")

