# Running the sweeps on a SLURM cluster

The sweep scripts contain no cluster-specific values. Everything that differs between
clusters is passed when submitting:

```bash
cd BaseAttacker && mkdir -p logs
PYTHON_MODULE=<module> VENV=<venv path> RESULTS_ROOT=<where runs go> \
  sbatch --account=<account> --array=<range> scripts/slurm/<sweep>.sh
```

| Variable | Meaning | Default |
|---|---|---|
| `--account` | your SLURM allocation | none: always pass it |
| `PYTHON_MODULE` | module to load before activating the venv | not loaded |
| `VENV` | virtual environment | `../venv` |
| `RESULTS_ROOT` | where runs are written | `results` |
| `TRAIN_TIMEOUT` | training stops itself after this, so scoring still runs | `21h` |
| `HF_PUBLISH=0` | do not upload finished runs to the Hugging Face Hub | uploads |
| `EXTRA_ARGS` | settings appended to every run (they override the sweep's own) | none |

`sbatch` passes the submitting shell's environment to the job, so exporting these
variables (or prefixing the command with them, as above) is enough.

Smoke-test a sweep before running it in full, with two short tasks:

```bash
EXTRA_ARGS="max_episodes=100 experiment=smoke" sbatch --account=<account> --time=00:20:00 --array=0-1 scripts/slurm/<sweep>.sh
```

First-time setup on an Alliance Canada cluster: `bash scripts/setup_alliance.sh <venv path>`.
To publish from jobs, log in once on the cluster: `hf auth login`.

## Clusters this has been run on

| | fir | killarney |
|---|---|---|
| Account | `def-<group>` | `aip-<group>` |
| `PYTHON_MODULE` | `python/3.12` | `python/3.12` |
| Code and venv | under `~/scratch` | under `/project/<account>/<user>/` |
| `RESULTS_ROOT` | default (`results/`) | `$SCRATCH/envpois/results` |
| Notes | compute nodes have internet | `sbatch`/`squeue`/`module` need a login shell (`bash -lc`); `/scratch` is NFS and purge-eligible |

Keep `--time` shorter than the time left before a maintenance reservation, or jobs
stay pending with reason `ReqNodeNotAvail`.
