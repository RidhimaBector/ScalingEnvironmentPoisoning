"""One naming convention for runs, used for local folders and for Hugging Face paths.

    <experiment>/<env>-<variant>__<victim>__<config>__seed<N>

    sweep-3x3x3/grid3d-5x5__sarsa__default__seed0
    sweep-3x3x3/ring-n12__qlearning__default__seed2
    sweep-3x3x3/lunarlander-kick1000__dqn__default__seed1
    ablation-4x4/grid3d-4x4__qlearning__td3__seed0

  experiment  what the run belongs to (a sweep or study); setting `experiment`
  env         registry name of the environment
  variant     the environment variant, derived from its constructor arguments
  victim      registry name of the victim algorithm
  config      label for the attacker configuration when an experiment compares
              several; setting `config_name` (default "default")
  seed        the run's seed

Fields are lower-case and use only [a-z0-9.-]; fields are separated by a double
underscore, so a name can always be split back into its parts.
"""

import re
from typing import Any, Dict

# Short, readable labels for the constructor arguments that define a variant.
_VARIANT_KEYS = {"n_states": "n", "initial_random": "kick"}


def _clean(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"\.0\b", "", text)                 # 1000.0 -> 1000
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-") or "default"


def variant_label(cfg: Dict[str, Any]) -> str:
    """'4x4' for a grid; otherwise the environment arguments, e.g. 'n12', 'kick1000'."""
    kwargs = dict(cfg.get("env_kwargs") or {})
    dims = kwargs.pop("grid_dimensions", None)
    if cfg["env"] == "grid3d":
        dims = dims or (4, 4)
        return f"{int(dims[0])}x{int(dims[1])}"
    if not kwargs:
        return "default"
    return "-".join(f"{_VARIANT_KEYS.get(k, _clean(k))}{_clean(v)}" for k, v in sorted(kwargs.items()))


def run_name(cfg: Dict[str, Any]) -> str:
    return "__".join([
        f"{_clean(cfg['env'])}-{variant_label(cfg)}",
        _clean(cfg["victim_algo"]),
        _clean(cfg.get("config_name") or "default"),
        f"seed{int(cfg['seed'])}",
    ])


def run_path(cfg: Dict[str, Any]) -> str:
    """'<experiment>/<run name>': the folder under results/ and the path on the Hub."""
    return f"{_clean(cfg.get('experiment') or 'adhoc')}/{run_name(cfg)}"


def parse_run_name(name: str) -> Dict[str, Any]:
    """Inverse of run_name()."""
    env_variant, victim, config, seed = name.split("__")
    env, variant = env_variant.split("-", 1)
    return {"env": env, "variant": variant, "victim_algo": victim, "config_name": config,
            "seed": int(seed.removeprefix("seed"))}
