"""Every experiment setting, with its default, in one place.

Override any of them on the command line:

    python main.py with env=ring victim_algo=sarsa max_episodes=3000 seed=1

The same dictionary is used when a saved run is reloaded (score_attack.py,
eval_attack.py), so a checkpoint is always rebuilt with the settings it was trained
with. Hyperparameters internal to the attacker and victim learners (network sizes,
learning rates of the tabular victims, ...) live in config/config_default.yaml.
"""

DEFAULT_CONFIG = {
    # --- run ---
    "seed": 0,
    "model_dir": None,  # where checkpoints go; None = results/run_<id>_<time>
    "max_episodes": 6000,  # attack episodes
    "max_timesteps": 15,  # attack steps per attack episode
    "eval_freq_episode": 20,  # save a numbered checkpoint every N episodes
    "keep_checkpoints": 3,  # how many numbered checkpoints to keep
    "log_timestep_every": 20,  # log per-step scalars to Sacred every Nth episode (0 = never)
    # --- victim environment (see registry.py for the names) ---
    "env": "grid3d",
    "grid_size": None,  # grid3d only: N for an N x N grid
    "env_shape": None,  # grid3d only: [rows, cols]
    "env_kwargs": None,  # constructor arguments for the environment class
    "env_args": "",  # same, from the command line: "n_states:12"
    # --- victim ---
    "victim_algo": "qlearning",
    "victim_n_episodes": 80,  # victim training episodes after each attack step
    "num_victims": 1,
    "victim_alpha": 0.1,  # learning rate of the tabular victims
    "victim_kwargs": None,  # constructor arguments for the victim class
    "victim_args": "",  # same, from the command line: "gamma:0.95"
    "privacy_mode": "full_blackbox",  # full_blackbox | full_whitebox
    # --- what the attacker observes ---
    # encoder_mode: "ae"   pre-trained autoencoder of the per-state behaviour trace
    #                      (tabular victims)
    #               "lstm" LSTM over the victim's state-action trajectories, trained
    #                      online (any victim; required for continuous states)
    #               "whitebox" reads the victim's policy table directly
    "encoder_mode": "ae",
    "ae_model_path": None,  # None = ae/models/<default for this environment>
    # obs_dynamics: what is appended to the behaviour embedding
    #   "actions" the running sum of the attacker's own perturbations (no read access
    #             to the environment)
    #   "env"     the environment's dynamics vector
    #   "none"    nothing
    "obs_dynamics": "actions",
    "normalize_obs": False,  # obs_dynamics="env": scale by the environment's bounds
    "lstm_append_dynamics": True,  # encoder_mode="lstm": append the obs_dynamics vector
    "lstm_embedding_dim": None,  # None = config/config_default.yaml LSTM_ENCODER values
    "lstm_seq_len": None,
    "lstm_memory_size": None,
    # --- attack reward (computed from observed behaviour only) ---
    #   "trace"     share of target states whose last observed action is the target's
    #   "traj"      share of observed actions in target states that are the target's
    #   "mix"       mean of the two
    #   "embedding" similarity of the victim's behaviour embedding to the target's,
    #               minus effort_weight x attack size (targets given as trajectories)
    "reward_mode": "trace",
    "effort_weight": 0.0,
    # --- attacker learner ---
    "attack_algo": "ddpg",  # ddpg | ppo
    "batch_size": 256,
    "discount": 0.9,
    "tau": 0.005,
    "attack_rate": 0.001,  # critic learning rate
    "attack_prate": 0.0001,  # actor learning rate
    "expl_noise": 0.1,  # std of Gaussian noise on the attacker's actions
    "updates_per_step": 15,  # gradient updates after each attack episode
    "eps_greedy_start_episodes": 30,  # warm-up episodes with random actions
    "td3": False,  # twin critics, target smoothing, delayed actor
    "bc_weight": 0.0,  # weight of imitating the elite_k best episodes
    "elite_k": 20,
    # --- push exploration ---
    # In a push episode every perturbation dimension is driven in one fixed direction
    # (down / hold / up) for the whole episode. With push_cem the directions are
    # sampled from probabilities that move towards the best push episodes.
    "explore_push_prob": 0.2,  # share of episodes that are push episodes (0 = off)
    "push_cem": True,
    "cem_batch": 16,  # push episodes per update
    "cem_elite": 4,  # best push episodes kept per update
    "cem_lr": 0.5,  # step towards the elite direction frequencies
    "cem_floor": -1.0,  # min probability per direction; < 0 = 0.25 / n_dims
}
