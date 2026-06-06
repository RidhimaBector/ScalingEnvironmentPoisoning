# Sacred experiment configuration for environment poisoning attack experiments.
#
# This module sets up Sacred for experiment tracking with MongoDB observer.
# All experiment runs, configurations, and metrics are automatically logged.
#
# Examples:
#     # Run with default config
#     python main.py
#
#     # Run with modified config
#     python main.py with max_episodes=50000 batch_size=512
#
#     # Run with specific MongoDB database
#     python main.py -m localhost:27017:env_poisoning_experiments

import os
import pkgutil
import importlib.util
import subprocess
import time

# Python 3.14 removed pkgutil.find_loader; patch it for sacred compatibility
if not hasattr(pkgutil, 'find_loader'):
    def _find_loader_compat(name):
        try:
            return importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            return None
    pkgutil.find_loader = _find_loader_compat

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

# Create the experiment
ex = Experiment('environment_poisoning_attack')


def _ensure_mongo_running(host='localhost', port=27017, timeout=15) -> bool:
    """Check if MongoDB is responsive; start it via brew if not."""
    import socket
    def _is_up():
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    if _is_up():
        return True

    print("MongoDB not running — starting via brew services...")
    try:
        subprocess.run(
            ['brew', 'services', 'start', 'mongodb-community'],
            check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not start MongoDB: {e}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_up():
            print("MongoDB started.")
            return True
        time.sleep(1)

    print("Warning: MongoDB did not become ready in time.")
    return False


def setup_observers(ex, mongo_url=None, db_name=None, use_file_observer=True):
    """
    Set up observers for the experiment.

    Args:
        ex: Sacred Experiment instance
        mongo_url: MongoDB connection URL (default: localhost:27017)
        db_name: Database name (default: env_poisoning)
        use_file_observer: Whether to also use file observer as backup
    """
    # MongoDB observer (primary)
    mongo_url = mongo_url or os.environ.get('SACRED_MONGO_URL', 'localhost:27017')
    db_name = db_name or os.environ.get('SACRED_DB_NAME', 'env_poisoning')

    mongo_connected = False
    if _ensure_mongo_running():
        try:
            ex.observers.append(MongoObserver(url=mongo_url, db_name=db_name))
            print(f"MongoDB observer connected: {mongo_url}/{db_name}")
            mongo_connected = True
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB ({e}). Using file observer only.")

    # File observer: only add as fallback when MongoDB is unavailable
    if use_file_observer and not mongo_connected:
        runs_dir = os.path.join(os.path.dirname(__file__), 'sacred_runs')
        os.makedirs(runs_dir, exist_ok=True)
        ex.observers.append(FileStorageObserver(runs_dir))


# Default configuration - mirrors config_default.yaml but exposed to Sacred
@ex.config
def default_config():
    """Default experiment configuration."""
    # Experiment settings
    experiment_name = "env_poisoning_attack"
    privacy_mode = "full_whitebox"  # full_whitebox | full_blackbox

    # Training parameters
    seed = 0
    max_episodes = 30000
    max_timesteps = 15
    eval_freq_episode = 20
    eps_greedy_start_episodes = 30
    victim_n_episodes = 80
    keep_checkpoints = 3  # number of recent model checkpoints to keep on disk

    # DDPG hyperparameters
    batch_size = 256
    discount = 0.9
    tau = 0.005
    attack_rate = 0.001    # critic learning rate
    attack_prate = 0.0001  # actor learning rate
    expl_noise = 0.1
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    # Autoencoder settings
    ae_embedding_size = 5
    ae_seq_len = 5
    ae_n_epochs = 10

    # Victim Q-learning settings
    victim_memory_size = 50
    victim_discount_factor = 1.0
    victim_alpha = 0.1
    victim_epsilon = 0.1

    # Model directory
    model_dir = None  # Will be auto-generated if None

    # Victim population
    num_victims = 3

    # Victim algorithm: "qlearning", "sarsa", or "reinforce"
    victim_algo = "qlearning"

    # Attack algorithm: "ddpg" (off-policy) or "ppo" (on-policy)
    attack_algo = "ddpg"

    # Victim environment: "grid3d" (default)
    env = "grid3d"

    # Optional grid shape as [rows, cols]. None = use env default (e.g. [4,4] for Grid3D).
    # Example: env_shape=[6,6] for a 6x6 grid.
    env_shape = None

    # Shorthand for square grids: grid_size=6 → (6,6). Overrides env_shape if set.
    grid_size = None

    # Observation encoder: "whitebox" (VictimEncoder: Q+dynamics+trace) or
    # "ae" (pre-trained AE on behavior_trace + altitude, replicates main branch)
    encoder_mode = "whitebox"


@ex.named_config
def whitebox():
    """Full whitebox attack configuration."""
    privacy_mode = "full_whitebox"


@ex.named_config
def blackbox():
    """Full blackbox attack configuration."""
    privacy_mode = "full_blackbox"


@ex.named_config
def quick_test():
    """Quick test configuration for debugging."""
    max_episodes = 100
    max_timesteps = 5
    eval_freq_episode = 10
    eps_greedy_start_episodes = 5


@ex.named_config
def long_training():
    """Extended training configuration."""
    max_episodes = 100000
    eval_freq_episode = 100


class SacredMetricsLogger:
    """
    Helper class to log metrics to Sacred during training.

    Usage:
        logger = SacredMetricsLogger(_run)
        logger.log_metric("accuracy", 0.95, step=100)
        logger.log_episode_metrics(episode=1, metrics_dict)
    """

    def __init__(self, run):
        """
        Initialize the metrics logger.

        Args:
            run: Sacred _run object passed to the main function
        """
        self._run = run
        self._step_counters = {}

    def log_metric(self, name, value, step=None):
        """
        Log a single metric value.

        Args:
            name: Metric name (e.g., "training.loss", "accuracy")
            value: Metric value
            step: Optional step number (auto-increments if not provided)
        """
        if step is None:
            step = self._step_counters.get(name, 0)
            self._step_counters[name] = step + 1

        self._run.log_scalar(name, value, step)

    def log_episode_metrics(self, episode, metrics_dict):
        """
        Log all metrics for an episode.

        Args:
            episode: Episode number
            metrics_dict: Dictionary of metric_name -> value
        """
        for name, value in metrics_dict.items():
            self._run.log_scalar(name, value, episode)

    def log_timestep_metrics(self, episode, timestep, metrics_dict):
        """
        Log metrics for a specific timestep within an episode.

        Args:
            episode: Episode number
            timestep: Timestep within episode
            metrics_dict: Dictionary of metric_name -> value
        """
        global_step = episode * 1000 + timestep  # Combine for unique step
        for name, value in metrics_dict.items():
            self._run.log_scalar(f"timestep.{name}", value, global_step)

    def log_artifact(self, filename):
        """
        Log a file as an artifact.

        Args:
            filename: Path to the file to log
        """
        self._run.add_artifact(filename)

    def log_info(self, key, value):
        """
        Log arbitrary info to the run's info dict.

        Args:
            key: Info key
            value: Info value
        """
        self._run.info[key] = value

    def log_collective_metrics(self, episode, metrics_dict, per_victim_accuracies=None):
        """
        Log metrics for collective (multi-victim) attack mode.

        Args:
            episode: Episode number
            metrics_dict: Dictionary of collective metrics (mean/min/max/std accuracy, etc.)
            per_victim_accuracies: Optional list of per-victim accuracy values
        """
        # Log collective metrics
        for name, value in metrics_dict.items():
            if not name.startswith('per_victim'):  # Skip per-victim lists
                self._run.log_scalar(f"collective.{name}", value, episode)

        # Log per-victim accuracies if provided
        if per_victim_accuracies is not None:
            for k, acc in enumerate(per_victim_accuracies):
                self._run.log_scalar(f"victim_{k}.accuracy", acc, episode)


# --- Dual experiment: victim tracking ---
#
# The victim experiment is a SEPARATE Sacred experiment instance that logs
# per-victim metrics (accuracy, behavior traces) independently from the
# attack experiment. This gives clean separation:
#
#   Attack Sacred (ex)  → DDPG loss, episode reward, mean accuracy, checkpoints
#   Victim Sacred (victim_ex) → per-victim accuracy, convergence, behavior stats
#
# Data flow:
#   VictimSystem.run_experiments()
#     → trains victims
#     → logs per-victim metrics to Victim Sacred (via VictimExperimentTracker)
#     → returns results to Encoder
#     → Encoder produces state for Attack Agent
#     → attack metrics logged to Attack Sacred (via AttackExperimentTracker)

victim_ex = Experiment('victim_tracking')


@victim_ex.config
def victim_default_config():
    """Default config for victim experiment."""
    experiment_name = "victim_tracking"
    num_victims = 1
    privacy_mode = "full_whitebox"
    victim_n_episodes = 80
    seed = 0


@victim_ex.main
def victim_main(_run, _config):
    """No-op main. The victim experiment is started manually via
    start_victim_sacred() and kept alive for the duration of training.
    The _run object is passed to VictimExperimentTracker for logging.
    """
    pass


def _setup_victim_observers():
    """Add observers to victim_ex (separate DB/file storage from attack)."""
    if victim_ex.observers:
        return  # Already configured

    mongo_url = os.environ.get('SACRED_MONGO_URL', 'localhost:27017')
    db_name = os.environ.get('SACRED_DB_NAME', 'env_poisoning')
    victim_db = f"{db_name}_victims"

    victim_mongo_connected = False
    if os.environ.get('SACRED_FILE_ONLY', '').lower() not in ('1', 'true', 'yes') and _ensure_mongo_running():
        try:
            victim_ex.observers.append(MongoObserver(url=mongo_url, db_name=victim_db))
            print(f"Victim MongoDB observer connected: {mongo_url}/{victim_db}")
            victim_mongo_connected = True
        except Exception as e:
            print(f"Warning: Victim MongoDB unavailable ({e})")

    # File observer: only add as fallback when MongoDB is unavailable
    if not victim_mongo_connected:
        runs_dir = os.path.join(os.path.dirname(__file__), 'sacred_runs_victims')
        os.makedirs(runs_dir, exist_ok=True)
        victim_ex.observers.append(FileStorageObserver(runs_dir))


def start_victim_sacred(attack_config):
    """Start the victim Sacred experiment and return a live _run object.

    Uses Sacred's _create_run() to get a Run object without blocking.
    The Run is started (observers notified) and stays open until
    finish_victim_sacred() is called.

    Args:
        attack_config: Dict from the attack experiment's _config.

    Returns:
        Sacred Run object, or None if startup fails.
    """
    _setup_victim_observers()

    try:
        config_updates = {
            'num_victims': attack_config.get('num_victims', 1),
            'privacy_mode': attack_config.get('privacy_mode', 'full_whitebox'),
            'victim_n_episodes': attack_config.get('victim_n_episodes', 80),
            'seed': attack_config.get('seed', 0),
        }
        run = victim_ex._create_run(
            config_updates=config_updates,
        )
        # _create_run doesn't set _output_file (normally set by run()).
        # The heartbeat thread needs .closed and .get(), so provide a stub.
        if run._output_file is None:
            class _DummyOutput:
                closed = False
                def get(self):
                    return ""
            run._output_file = _DummyOutput()
        # Notify observers that the experiment has started
        run._emit_started()
        # Start heartbeat so metrics are periodically flushed to file observer
        run._start_heartbeat()
        print(f"Victim Sacred experiment started: run_id={run._id}")
        return run

    except Exception as e:
        print(f"Warning: Could not start victim Sacred experiment ({e}). "
              "Victim metrics will not be logged to Sacred.")
        return None


def finish_victim_sacred(run, status='COMPLETED'):
    """Signal that the victim Sacred experiment is complete.

    Args:
        run: Sacred Run object from start_victim_sacred().
        status: Final status ('COMPLETED' or 'FAILED').
    """
    if run is None:
        return
    try:
        import datetime
        # Flush pending metrics via a final heartbeat
        run._emit_heartbeat()
        run._stop_heartbeat()
        # Sacred observers expect stop_time to be set
        run.stop_time = datetime.datetime.now()
        if status == 'COMPLETED':
            run._emit_completed(result=None)
        else:
            run._emit_failed(fail_trace=status)
        print(f"Victim Sacred experiment finished: {status}")
    except Exception as e:
        print(f"Warning: Error finishing victim Sacred experiment: {e}")


def create_dual_experiments(mongo_url=None, db_name=None):
    """Create and configure both attack and victim Sacred experiments.

    Returns:
        Tuple of (attack_experiment, victim_experiment).
    """
    mongo_url = mongo_url or os.environ.get('SACRED_MONGO_URL', 'localhost:27017')
    db_name = db_name or os.environ.get('SACRED_DB_NAME', 'env_poisoning')

    setup_observers(ex, mongo_url, db_name)
    _setup_victim_observers()

    return ex, victim_ex


# Initialize observers when module is imported
# Can be customized via environment variables:
#   SACRED_MONGO_URL=localhost:27017
#   SACRED_DB_NAME=env_poisoning
#   SACRED_FILE_ONLY=1  (to disable MongoDB)
if os.environ.get('SACRED_FILE_ONLY', '').lower() not in ('1', 'true', 'yes'):
    setup_observers(ex)
else:
    # File observer only
    runs_dir = os.path.join(os.path.dirname(__file__), 'sacred_runs')
    os.makedirs(runs_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(runs_dir))
    print(f"Using file observer only: {runs_dir}")
