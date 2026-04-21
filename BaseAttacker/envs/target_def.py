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

    # For 4x4 grid with 4 actions (specific target paths)
    if nS == 16 and nA == 4:
        if path_type == "Mp":
            # Mp target path: right along top, down right side, left along bottom
            # Actions: 0=UP, 1=RIGHT/EAST, 2=DOWN/SOUTH, 3=LEFT/WEST
            target[0][1] = 1   # State 0: EAST
            target[1][1] = 1   # State 1: EAST
            target[2][1] = 1   # State 2: EAST
            target[3][2] = 1   # State 3: SOUTH
            target[7][2] = 1   # State 7: SOUTH
            target[11][2] = 1  # State 11: SOUTH
            target[15][3] = 1  # State 15: WEST
            target[14][3] = 1  # State 14: WEST
            target[13][3] = 1  # State 13: WEST
        elif path_type == "M" or path_type == "H":
            # M and H target paths (partial)
            target[0][1] = 1   # State 0: EAST
            target[1][1] = 1   # State 1: EAST
            target[2][1] = 1   # State 2: EAST
            target[3][2] = 1   # State 3: SOUTH
            target[7][2] = 1   # State 7: SOUTH
            target[11][2] = 1  # State 11: SOUTH
        elif path_type == "E":
            # E target path (down left side, right along bottom)
            target[0][2] = 1   # State 0: SOUTH
            target[4][2] = 1   # State 4: SOUTH
            target[8][2] = 1   # State 8: SOUTH
            target[12][1] = 1  # State 12: EAST
            target[13][1] = 1  # State 13: EAST
    else:
        # Generic fallback: for each state, prefer action 0 with small prob on others
        for s in range(nS):
            target[s] = np.ones(nA) * 0.001
            target[s][0] = 1.0
            target[s] = target[s] / np.sum(target[s])

    return target


# Default target for backward compatibility (used by victim_Q.py)
TARGET = create_target_policy(16, 4, "Mp")

MEM_Target = Memory(MEMORY_SIZE)
