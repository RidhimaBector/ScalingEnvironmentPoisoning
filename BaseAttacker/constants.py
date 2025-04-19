#SEQ_LEN = 6 #config.AE.SEQ_LEN
EMBEDDING_SIZE = 5 #config.AE.EMBEDDING_SIZE
MEMORY_SIZE = 50 #config.AE.MEMORY_SIZE

WHITEBOX_METRICS = ["distance_K", "distance_grid_K", "distance_behavior_K", "distance_W", "distance_grid_W", "distance_behavior_W", "accuracy", "accuracy_sftmx", "accuracy_sftmx_complete", "effort", "time"]
BLACKBOX_METRICS = ["policy_change", "trajectory_change", "accuracy", "accuracy_sftmx", "accuracy_sftmx_complete", "effort", "time"]
PARTIAL_BLACKBOX_METRICS = ["embedding_distance", "cosine_similarity", "accuracy", "accuracy_sftmx", "accuracy_sftmx_complete", "effort", "time"]


VICTIM_ALGO_KWARGS = {
    "memory_size": MEMORY_SIZE,
    "discount_factor": 0.9, #1.0,
    "alpha": 0.1,
    "epsilon": 0.1,
}
DEFAULT_AE_KWARGS = {
    "enc_in_size": 32, #SEQ_LEN*2
    "enc_out_size": 5, #EMBEDDING_SIZE
    "dec_in_size": 6, #1+EMBEDDING_SIZE
    "dec_out_size": 5, #4 #attack_action_dim
    "lr": 0.001, #0.001,
}

