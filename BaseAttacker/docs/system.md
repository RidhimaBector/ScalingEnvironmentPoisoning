# System Classes Implementation

This file documents the system classes that manage different agents and their environments in the attack framework.

## Base System Class

### Overview
The `System` class serves as an abstract base class that defines the core interface for managing agent-environment interactions.

### Key Components
- Abstract base class implementation
- Environment and algorithm management
- Type hints for better code clarity

### Main Methods
- `__init__(env, algorithm)`: Initializes with environment and algorithm instances
- `reset()`: Abstract method to reset environment and algorithm state
- Property getters for environment and algorithm access

## Victim System Class

### Overview
The `VictimSystem` class manages the victim agent's training and evaluation process, extending the base System class.

### Key Components
- Specialized for victim environment management
- Handles victim agent training cycles
- Maintains victim's policy state

### Main Methods
- `__init__(env, algorithm)`: Initializes victim system with specific environment
- `reset()`: Resets victim environment altitude and algorithm state

## Attack System Class

### Overview
The `AttackSystem` class manages the attack agent's operations and metrics tracking, implementing the most complex system functionality.

### Key Components
- Attack environment management
- Autoencoder integration
- Comprehensive metrics tracking
- Training and evaluation capabilities

### Main Methods

#### Core System Operations
- `__init__(env, algorithm, env_autoencoder)`: Initializes attack system with environment, algorithm, and autoencoder
- `reset()`: Resets attack environment and algorithm state
- `select_action(state, episode)`: Handles action selection with epsilon-greedy strategy

#### Training Functions
- `train_victim()`: Manages victim agent training cycles
- `train_system()`: Implements main training loop
- `run_training_episode()`: Executes single training episode
- `_run_training_step()`: Performs individual training step

#### Attack Metrics
- `attack_effort()`: Calculates attack effort metrics
- `attack_cost_compute_K()`: Computes KL divergence-based cost
- `attack_cost_compute_W()`: Calculates Wasserstein distance-based cost
- `attack_done_identify()`: Determines attack completion status

#### Metrics and Statistics
- `calculate_attack_metrics()`: Computes comprehensive attack metrics
- `update_model_statistics()`: Updates best/worst model tracking
- `_compute_kl_distance()`: Calculates KL divergence metrics
- `_compute_wasserstein_distance()`: Calculates Wasserstein distance metrics

### Metrics Tracked
- KL divergence distances
- Wasserstein distances
- Accuracy metrics
- Attack effort
- Model performance statistics

### Data Management
The system maintains several data structures:
- Model performance data
- Buffer metrics
- Best/worst model statistics
- Environment dynamics tracking

## Usage Example

```python
# Initialize systems
victim_system = VictimSystem(victim_env, victim_algorithm)
attack_system = AttackSystem(attack_env, attack_algorithm, autoencoder)

# Training loop
for episode in range(max_episodes):
    # Reset systems
    victim_system.reset()
    attack_system.reset()

    # Run training episode
    metrics = attack_system.run_training_episode(
        episode, victim_system, buffer, max_timesteps
    )
```

## Implementation Notes
- All systems implement the abstract base class interface
- Type hints are used throughout for better code clarity
- Metrics tracking is centralized in the AttackSystem class
- Environment state management is handled consistently across systems
