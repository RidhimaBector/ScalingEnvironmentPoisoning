import os
import sys
from os.path import abspath, dirname

import numpy as np

from utils.utils_buf import Memory

if "../" not in sys.path:
    sys.path.append("../")

from yacs.config import CfgNode as CN

yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

MEMORY_SIZE = config.AE.MEMORY_SIZE


def create_target_policy(nS: int, nA: int, path_type: str = "Mp") -> np.ndarray:
    """Create a target policy matrix.

    Args:
        nS: Number of states in the environment
        nA: Number of actions in the environment
        path_type: Type of target path ("Mp", "M", "H", "E")

    Returns:
        Target policy matrix of shape (nS, nA)
    """
    target = np.zeros((nS, nA))

    # Infer grid shape from nS (assumes square or near-square grid)
    ncols = int(round(nS ** 0.5))
    nrows = nS // ncols

    if nA >= 4 and nrows * ncols == nS and path_type in ("Mp", "M", "H", "E"):
        # Actions: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
        if path_type == "Mp":
            # Perimeter path: top row east → right column south → bottom row west → goal (bottom-left)
            # Top row (except top-right corner): EAST
            for c in range(ncols - 1):
                target[c][1] = 1
            # Right column (except top-right corner): SOUTH
            for r in range(nrows - 1):
                target[r * ncols + (ncols - 1)][2] = 1
            # Bottom row (except bottom-left corner, which is the goal): WEST
            for c in range(ncols - 1, 0, -1):
                target[(nrows - 1) * ncols + c][3] = 1
        elif path_type in ("M", "H"):
            # Top row (except top-right corner): EAST
            for c in range(ncols - 1):
                target[c][1] = 1
            # Right column (except corners): SOUTH
            for r in range(nrows - 1):
                target[r * ncols + (ncols - 1)][2] = 1
        elif path_type == "E":
            # Left column (except bottom-left corner): SOUTH
            for r in range(nrows - 1):
                target[r * ncols][2] = 1
            # Bottom row (except bottom-left corner): EAST
            for c in range(1, ncols):
                target[(nrows - 1) * ncols + c][1] = 1
    else:
        # Generic fallback: uniform preference for action 0
        for s in range(nS):
            target[s][0] = 1.0

    return target


# Default target for backward compatibility (used by victim_Q.py)
TARGET = create_target_policy(16, 4, "Mp")

MEM_Target = Memory(MEMORY_SIZE)
