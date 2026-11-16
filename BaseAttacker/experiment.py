"""Sacred experiment tracking for attack runs.

Two Sacred runs are recorded per training run:

  attack run   the attacker's side: episode reward, losses, checkpoints, and the
               ground-truth success metrics used for evaluation
  victim run   the victims' side: each victim's accuracy or return after every
               relearning phase

Nothing in training reads these back; they are records only.

Where they go:
  default                      files under sacred_runs/ and sacred_runs_victims/
  SACRED_MONGO_URL=host:port   a MongoDB instead (databases <SACRED_DB_NAME> and
                               <SACRED_DB_NAME>_victims, default "env_poisoning")
"""

import datetime
import importlib.util
import os
import pkgutil

# Python 3.14 removed pkgutil.find_loader, which Sacred still calls.
if not hasattr(pkgutil, "find_loader"):

    def _find_loader_compat(name):
        try:
            return importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            return None

    pkgutil.find_loader = _find_loader_compat

from sacred import Experiment  # noqa: E402
from sacred.observers import FileStorageObserver, MongoObserver  # noqa: E402
from sacred.run import Run as _SacredRun  # noqa: E402

from defaults import DEFAULT_CONFIG  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sacred's FileStorageObserver re-reads and rewrites the whole metrics.json on every
# heartbeat (every 10 s by default). Flush once a minute instead.
_orig_start_heartbeat = _SacredRun._start_heartbeat


def _start_heartbeat_slow(self):
    self.beat_interval = float(os.environ.get("SACRED_BEAT_INTERVAL", 60))
    return _orig_start_heartbeat(self)


_SacredRun._start_heartbeat = _start_heartbeat_slow


def _add_observer(experiment: Experiment, db_suffix: str, file_dir: str) -> None:
    """MongoDB if SACRED_MONGO_URL is set and reachable, otherwise files."""
    mongo_url = os.environ.get("SACRED_MONGO_URL")
    if mongo_url:
        db_name = os.environ.get("SACRED_DB_NAME", "env_poisoning") + db_suffix
        try:
            experiment.observers.append(MongoObserver(url=mongo_url, db_name=db_name))
            return
        except Exception as exc:  # unreachable server, missing pymongo, ...
            print(
                f"Warning: MongoDB at {mongo_url} unavailable ({exc}); logging to files."
            )
    path = os.path.join(_HERE, file_dir)
    os.makedirs(path, exist_ok=True)
    experiment.observers.append(FileStorageObserver(path))


# --- attack run -------------------------------------------------------------

ex = Experiment("environment_poisoning_attack")
ex.add_config(DEFAULT_CONFIG)
_add_observer(ex, "", "sacred_runs")


@ex.named_config
def quick_test():
    """A run that finishes in seconds, to check an installation."""
    max_episodes = 5  # noqa: F841
    max_timesteps = 3  # noqa: F841
    victim_n_episodes = 5  # noqa: F841
    eps_greedy_start_episodes = 2  # noqa: F841
    batch_size = 4  # noqa: F841


@ex.named_config
def lander():
    """LunarLander with a DQN victim: trajectory encoder and embedding reward."""
    env = "lunarlander"  # noqa: F841
    victim_algo = "dqn"  # noqa: F841
    encoder_mode = "lstm"  # noqa: F841
    reward_mode = "embedding"  # noqa: F841
    effort_weight = 0.1  # noqa: F841
    lstm_seq_len = 200  # noqa: F841
    lstm_memory_size = 4000  # noqa: F841
    lstm_embedding_dim = 100  # noqa: F841
    victim_n_episodes = 20  # noqa: F841
    max_timesteps = 10  # noqa: F841
    max_episodes = 1000  # noqa: F841


@ex.named_config
def whitebox():
    """The attacker reads the victim's policy table directly (upper reference)."""
    privacy_mode = "full_whitebox"  # noqa: F841
    encoder_mode = "whitebox"  # noqa: F841


# --- victim run -------------------------------------------------------------

victim_ex = Experiment("victim_tracking")
victim_ex.add_config(
    {
        "num_victims": 1,
        "privacy_mode": "full_blackbox",
        "victim_n_episodes": 80,
        "seed": 0,
    }
)


@victim_ex.main
def _victim_main():
    """Never called: the victim run is opened and closed by hand (below)."""


class _NoOutput:
    """Stands in for Sacred's captured-output file, which a hand-started run lacks."""

    closed = False

    def get(self):
        return ""


def start_victim_sacred(attack_config: dict):
    """Open the victim run and return its Sacred Run object (None if that fails)."""
    if not victim_ex.observers:
        _add_observer(victim_ex, "_victims", "sacred_runs_victims")
    try:
        run = victim_ex._create_run(
            config_updates={
                k: attack_config[k]
                for k in ("num_victims", "privacy_mode", "victim_n_episodes", "seed")
            }
        )
        if run._output_file is None:
            run._output_file = _NoOutput()
        run._emit_started()
        run._start_heartbeat()
        return run
    except Exception as exc:
        print(f"Warning: victim metrics will not be logged to Sacred ({exc}).")
        return None


def finish_victim_sacred(run, status: str = "COMPLETED") -> None:
    """Close the victim run opened by start_victim_sacred()."""
    if run is None:
        return
    try:
        run._emit_heartbeat()
        run._stop_heartbeat()
        run.stop_time = datetime.datetime.now()
        if status == "COMPLETED":
            run._emit_completed(result=None)
        else:
            run.fail_trace = [status]
            run._emit_failed()
    except Exception as exc:
        print(f"Warning: error closing the victim Sacred run: {exc}")
