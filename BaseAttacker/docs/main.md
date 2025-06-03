# Environment Poisoning Attack Implementation

This file implements a framework for training an attacker agent to poison a reinforcement learning environment. The attacker attempts to modify the environment dynamics to influence the victim agent's learned policy.

## Key Components

### Environment Setup
- Uses a 3D Grid (4x4) environment
- Implements both victim and attack environments
- Utilizes an autoencoder for state representation

### Main Classes/Systems
- `VictimSystem`: Manages the victim agent and its environment
- `AttackSystem`: Manages the attack agent and its interactions
- `AttackEnvironment`: Special environment for the attacker's actions

### Attack Implementation
The attack is implemented using DDPG (Deep Deterministic Policy Gradient) with the following features:
- Replay buffer for experience storage
- Epsilon-greedy exploration strategy
- Periodic model saving and evaluation

## Key Functions

### `_create_cost_matrix(victim_env)`
Creates a cost matrix for the attack environment using cityblock distance metric.

### `parse_arguments()`
Handles command-line arguments including:
- Model directory settings
- Training parameters (episodes, timesteps, batch size)
- DDPG hyperparameters
- Evaluation frequency

### `_set_seeds(seed, victim_env, attack_env)`
Sets random seeds for reproducibility across:
- Victim environment
- Attack environment
- PyTorch
- NumPy

### `main()`
Main training loop that:
1. Initializes environments and agents
2. Sets up the autoencoder
3. Implements the training loop with:
   - Episode-based training
   - Metrics tracking
   - Model saving
   - Performance evaluation

## Usage

```bash
python main.py [arguments]
```

### Key Arguments
- `--model_dir`: Directory for saving models
- `--max_episodes`: Maximum training episodes
- `--max_timesteps`: Maximum timesteps per episode
- `--batch_size`: Batch size for training
- `--discount`: Discount factor for DDPG
- `--eval_freq_episode`: Frequency of evaluation

## Metrics Tracked
- Distance metrics (KL divergence and Wasserstein)
- Accuracy metrics
- Attack effort
- Execution time

## Model Saving
The implementation saves:
- Buffer states
- Policy models
- DDPG loss
- Various performance metrics
- Best and worst performing models
