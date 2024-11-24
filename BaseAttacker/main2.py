import numpy as np
import torch
import argparse
import os
import time
import copy
import ot
import datetime
from utils import utils_buf, utils_op, utils_attack, utils_log
from attack.DDPG import DDPG
from agent.agent import VictimAgent, AttackAgent
from constants import *
from envs.env3D_4x4 import GridWorld_3D_env
from envs.target_def import TARGET
from envs.environment import AttackEnvironment
from victim.victim_Q import VictimQLearning
from ae.ae import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_cost_matrix(victim_env):
    """Create a cost matrix for the attack environment."""
    grid = np.mgrid[0:4, 0:4].reshape(2,-1).T
    cost_matrix = ot.dist(grid, grid, metric='cityblock') * 20
    np.fill_diagonal(cost_matrix, 10)
    cost_matrix = np.tile(cost_matrix, (victim_env.nA, victim_env.nA))
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="R_0818")               # TensorBoard folder
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--atk_training_start_episodes", default=100, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eps_greedy_start_episodes", default=30, type=int)  # Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=15, type=int)   # Max time steps to run environment
    parser.add_argument("--max_episodes_num", default=30000, type=int)   # Max episodes to run environment
    parser.add_argument("--eval_freq_episode", default=20, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.9, type=float)     # Discount factor of attack network $\gamma$DDPG with Fixed Discount
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    parser.add_argument("--victim_n_episodes", default=80, type=int)  # number of episodes for victim's updated in poisoned Env
    parser.add_argument("--ae_n_epochs", default=10, type=int)         # number of training epoch

    return parser.parse_args()


def _set_seeds(seed, victim_env, attack_env):
    victim_env.seed(seed)
    attack_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _save_attack_policy(model_dir, no_episodes, Buffer, Policy, ddpg_loss, buffer_metrics, model_data, model_good_data, model_bad_data):
    Buffer.saveBuffer(f"./{model_dir}/")
    Policy.save(f"./{model_dir}/{no_episodes}")
    np.savetxt("ddpg_loss.csv", np.array(ddpg_loss), delimiter=",")
    for metric in METRICS:
        np.savetxt(f"{metric}_buffer.csv", np.array(buffer_metrics[metric]), delimiter=",")
    np.savetxt("model_data.csv", np.array(model_data), delimiter=",")
    np.savetxt("model_good_data.csv", np.array(model_good_data), delimiter=",")
    np.savetxt("model_bad_data.csv", np.array(model_bad_data), delimiter=",")


def main():
    args = parse_arguments()
    victim_env = GridWorld_3D_env()
    attack_env = AttackEnvironment(victim_env)
    INIT_T = victim_env.T.copy()

    _set_seeds(args.seed, victim_env, attack_env)
    cost_matrix = _create_cost_matrix(victim_env)

    ''' ..... Tensorboard Settings ..... '''

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"results_{date}"

    model_name = args.model_dir or default_model_name
    model_dir = utils_log.get_model_dir(model_name)
    os.makedirs(model_dir, exist_ok=True)

    ''' ..... Attack Network ..... '''
    # Input / Output size
    attack_state_dim = attack_env.nS
    attack_action_dim = attack_env.action_space.shape[0]
    max_action = float(attack_env.action_space.high[0])

    kwargsNew = {
        "seed": args.seed,
        "nb_states": attack_state_dim,
        "nb_actions": attack_action_dim,
        "max_action": max_action,
        "hidden1": 400,
        "hidden2": 300,
        "init_w": 0.003,
        "prate": 0.0001,
        "rate": 0.001,
        "ou_theta": 0.15,
        "ou_mu": 0.0,
        "ou_sigma": 0.2,
        "bsize": 256,
        "tau": args.tau,
        "discount": args.discount,
        "epsilon_divisor": 1000, #Number of episodes over which to decrease actual epsilon
        "is_training": True
    }

    # Initialize Policy -- attacker
    attack_algo = DDPG(**kwargsNew)
    Buffer = utils_buf.ReplayBuffer(attack_state_dim, attack_action_dim, max_size=int(1e6))

    ''' ..... Victim ..... '''
    victim_args = {
        "env": victim_env,
        "memory_size": MEMORY_SIZE,
        "discount_factor": 0.9, #1.0,
        "alpha": 0.1,
        "epsilon": 0.1,
    }

    victim_algo = VictimQLearning(**victim_args)
    victim_agent = VictimAgent(victim_env, victim_algo)
    attacker_agent = AttackAgent(attack_env, attack_algo)

    ae_enc_in_size = 32 #SEQ_LEN*2
    ae_enc_out_size = 5 #EMBEDDING_SIZE
    ae_dec_in_size = 6 #1+EMBEDDING_SIZE
    ae_dec_out_size = 5 #4 #attack_action_dim

    ae_args = {
        "enc_in_size": ae_enc_in_size,
        "enc_out_size": ae_enc_out_size,
        "dec_in_size": ae_dec_in_size,
        "dec_out_size": ae_dec_out_size,
        "lr": 0.001, #0.001,
    }

    ae = AutoEncoder(**ae_args)
    ae.load("ae/models/" + "340240" + "_f-o_AutoEncoder_SftMx") #"14800" + "_f-o_AutoEncoder") #44780 - p-o M/H
    #system = System(victim_args, ae_args)

    atk_n_epoch = 1 #15
    atk_n_batch = 15
    ddpg_loss = []
    model_data = [] #per episode statistic wrt all metrics
    model_good_data = [] #all model statistic for best models
    model_bad_data = [] #all model statistic for worst models

    buffer_metrics = {metric: [] for metric in METRICS} # complete data

    best_model_stats = np.zeros((33,1)) #best statistics so far (across diff models)
    worst_model_stats = np.zeros((33,1)) #worst statistics so far (across diff models)


    ''' ..... Training ..... '''
    for i_episode in range(args.max_episodes_num):
        print(f"\n--------- Episode: {i_episode} ----------") #txt_logger.info(f"\n--------- Episode: {i_episode} ----------")

        cumulative_metrics = temp_metrics = {metric: 0 for metric in METRICS}

        # reset victim's env and Q
        victim_env, victim_algo = victim_agent.reset()

        attack_env.victim = (victim_algo)
        attack_env.victim_env = (victim_env)
        attack_env.reset()

        x = attack_env.get_initial_state()
        curA = victim_env.altitude.copy().reshape((16, 1))

        for t in range(args.max_timesteps):
            tic_timestep = time.time()

            # Select attack_action
            if i_episode < args.eps_greedy_start_episodes:
                u = attack_algo.select_random_action(attack_env.action_space)
            else:
                u = attack_algo.act(np.array(x))

            # Step: implement attack_action
            attack_env.step(u)

            # Step: victim updates = get next_x
            victim_transitions = victim_algo.Train_Model(num_episodes=80)

            attack_env.victim = (victim_algo)
            attack_env.victim_env = (victim_env)
            ### ... updated victim.Q
            next_victim_info = ae.Policy_Embedding(victim_transitions) #next_victim_info = system.ae.Embedding(system.victim.MEM)
            next_victim_tensor = torch.from_numpy(next_victim_info[-1]).unsqueeze(0)
            next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)

            next_env_info = victim_env.altitude.copy() #system.victim.env.altitude.copy()
            next_env_tensor = torch.from_numpy(next_env_info)
            next_env_tensor = next_env_tensor.view(1, victim_env.nS)
            next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)

            next_x = attack_env.get_next_state(ae, victim_transitions)

            # Step: cost
            temp_metrics["distance_K"] = - utils_attack.Attack_Cost_Compute_K(victim_env, INIT_T, victim_algo.Q, TARGET, cost_matrix, distance_type=0) #system.victim.Q
            temp_metrics["distance_grid_K"] = - utils_attack.Attack_Cost_Compute_K(victim_env, INIT_T, victim_algo.Q, TARGET, cost_matrix, distance_type=1)
            temp_metrics["distance_behavior_K"] = - utils_attack.Attack_Cost_Compute_K(victim_env, INIT_T, victim_algo.Q, TARGET, cost_matrix, distance_type=2)
            temp_metrics["distance_W"] = - utils_attack.Attack_Cost_Compute_W(victim_env, INIT_T, victim_algo.Q, TARGET, cost_matrix, distance_type=0) #system.victim.Q
            temp_metrics["distance_grid_W"] = - utils_attack.Attack_Cost_Compute_W(victim_env, INIT_T, victim_algo.Q, TARGET, cost_matrix, distance_type=1)
            temp_metrics["distance_behavior_W"] = - utils_attack.Attack_Cost_Compute_W(victim_env, INIT_T, victim_algo.Q, TARGET, cost_matrix, distance_type=2)
            done, temp_metrics["accuracy"], temp_metrics["accuracy_sftmx"], temp_metrics["accuracy_sftmx_complete"] = utils_attack.Attack_Done_Identify(victim_env, TARGET, victim_algo.Q) #system.victim.Q)
            temp_metrics["effort"], curA = utils_attack.Attack_Effort(curA, victim_env)
            temp_metrics["effort"] = - temp_metrics["effort"]
            toc_timestep = time.time()
            temp_metrics["time"] = tic_timestep - toc_timestep #- (toc_timestep - tic_timestep)

            ### log
            no_episodes = i_episode+1
            no_timesteps = t+1
            for metric in METRICS:
                buffer_metrics[metric].append([no_episodes, no_timesteps, temp_metrics[metric]])

            reward = temp_metrics["accuracy"]
            for metric in temp_metrics:
                cumulative_metrics[metric] += temp_metrics[metric]

            # Replay buffer
            Buffer.add(x.view(attack_state_dim), u, next_x.view(attack_state_dim), reward, done)

            # Update state
            x = copy.deepcopy(next_x)
            if (done or (no_timesteps == args.max_timesteps)):

                stats = []
                for metric in METRICS:
                    stats.extend([
                        temp_metrics[metric],
                        cumulative_metrics[metric]/no_timesteps,
                        cumulative_metrics[metric]
                    ])
                cur_model_stats = np.array(stats)

                model_data.append([no_episodes, no_timesteps])
                model_data[-1].extend(cur_model_stats.tolist())

                if(i_episode == 0):
                    best_model_stats = copy.deepcopy(cur_model_stats)
                    worst_model_stats = copy.deepcopy(cur_model_stats)
                    break

                cur_model_is_good = cur_model_stats >= best_model_stats
                cur_model_is_bad = cur_model_stats <= worst_model_stats

                if(np.sum(cur_model_is_good) > 0):
                    model_good_data.append([no_episodes, no_timesteps])
                    model_good_data[-1].extend(cur_model_stats.tolist())
                    model_good_data[-1].extend(best_model_stats.tolist())
                    model_good_data[-1].extend(worst_model_stats.tolist())
                    attack_algo.save(f"./{model_dir}/good_model_{no_episodes}")

                    best_model_stats[cur_model_is_good] = cur_model_stats[cur_model_is_good]

                elif(np.sum(cur_model_is_bad) > 0):
                    model_bad_data.append([no_episodes, no_timesteps])
                    model_bad_data[-1].extend(cur_model_stats.tolist())
                    model_bad_data[-1].extend(best_model_stats.tolist())
                    model_bad_data[-1].extend(worst_model_stats.tolist())
                    attack_algo.save(f"./{model_dir}/bad_model_{no_episodes}")

                    worst_model_stats[cur_model_is_bad] = cur_model_stats[cur_model_is_bad]

                break

        # Attack_attack_algo Update
        if i_episode >= args.eps_greedy_start_episodes:
            ddpg_loss = attack_algo.train(Buffer, atk_n_epoch, atk_n_batch, args.batch_size, ddpg_loss, i_episode)

        ''' save Attack policy '''
        if no_episodes % args.eval_freq_episode == 0:
            _save_attack_policy(model_dir, no_episodes, Buffer, attack_algo, ddpg_loss, buffer_metrics, model_data, model_good_data, model_bad_data)


def idealMain():
    args = parse_arguments()

    victim_env = GridWorld_3D_env()
    victim_algo = VictimQLearning()
    victim_agent = VictimAgent(victim_env, victim_algo)

    attack_env = AttackEnvironment(victim_env)
    attack_algo = DDPG()
    attack_agent = AttackAgent(attack_env, attack_algo)

    buffer = utils_buf.ReplayBuffer()
    buffer_metrics = {metric: [] for metric in METRICS} # complete data

    ae = AutoEncoder()

    for i_episode in range(args.max_episodes_num):
        print(f"\n Episode: {i_episode}")
        cumulative_metrics = temp_metrics = {metric: 0 for metric in METRICS}
        victim_env, victim_algo = victim_agent.reset()
        attack_env, attack_algo = attack_agent.reset()




if __name__ == "__main__":
    main()
