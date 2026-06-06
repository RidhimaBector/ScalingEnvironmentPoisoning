#!/bin/bash
SEED=${1:-0}
python main.py with encoder_mode=ae num_victims=1 attack_algo=ddpg victim_algo=qlearning env=grid3d max_episodes=12000 seed=$SEED
