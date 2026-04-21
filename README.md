# ScalingEnvironmentPoisoning

Environment poisoning attacks against RL agents. Implements the attacker from:

2. Create virtual environment 
    1. Open Anaconda Prompt
    2. Run command “conda create -n envPois python=3.10” to create environment with name “envPois”.
    3. Run command “conda activate envPois” to activate the environment

3. Install required packages in the virtual environment
    1. conda install spyder
    2. conda install -c pytorch pytorch
    3. conda install -c conda-forge numpy
    4. conda install -c conda-forge gym
    5. conda install -c conda-forge pot
    6. conda install -c conda-forge tensorboardx
    7. conda install -c conda-forge yacs

> "Spiking Pitch Black: Poisoning an Unknown Environment to Attack Unknown Reinforcement Learners" (AAMAS 2022)

## Setup


Requires Python 3.10+ and MongoDB.

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
cd BaseAttacker
pip install -r requirements.txt
```

### 2. MongoDB

```bash
# macOS
brew tap mongodb/brew && brew install mongodb-community
brew services start mongodb-community

# Linux (Ubuntu/Debian)
sudo apt install mongodb
sudo systemctl start mongodb

# Docker (any platform)
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

> **No MongoDB?** Set `SACRED_FILE_ONLY=1` to save results to `sacred_runs/` instead.

## Run

```bash
cd BaseAttacker

# Train with defaults (whitebox encoder, 3 victims, Q-learning, DDPG)
python main.py

# Common overrides
python main.py --max_episodes 30000 --model_dir results/my_run
python main.py --victim_algo sarsa
python main.py --num_victims 5
python main.py --attack_algo ppo
python main.py --encoder legacy --privacy_mode full_blackbox
```

## Key flags

| Flag | Default | Options |
|------|---------|---------|
| `--encoder` | `whitebox` | `whitebox`, `legacy` |
| `--victim_algo` | `qlearning` | `qlearning`, `sarsa`, `reinforce` |
| `--attack_algo` | `ddpg` | `ddpg`, `ppo` |
| `--num_victims` | `3` | any int |
| `--max_episodes` | `30000` | any int |
| `--victim_n_episodes` | `80` | any int |
| `--privacy_mode` | `partial_blackbox` | `full_whitebox`, `partial_blackbox`, `full_blackbox` |

## GPU

If you get GPU memory errors, set `device = "cpu"` in `main.py`, `ae/ae.py`, and `attack/DDPG.py`.

## Experiment tracking

Experiments are tracked with [Sacred](https://sacred.readthedocs.io) and logged to MongoDB (`localhost:27017/env_poisoning`). View results with [Omniboard](https://github.com/vivekratnavel/omniboard):

```bash
npm install -g omniboard
omniboard -m localhost:27017:env_poisoning
# open http://localhost:9000
```
