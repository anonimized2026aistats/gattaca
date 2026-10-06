from typing import List, Tuple

import numpy as np

INPUT_MASK = np.ndarray
TRUTH_TABLE = np.ndarray
NODE_NAME = str
CONTROL_FLAG = bool

PBN_DATA = List[Tuple[INPUT_MASK, TRUTH_TABLE, NODE_NAME, CONTROL_FLAG]]

LOGIC_FUNC = List[Tuple[str, float]]
LOGIC_FUNC_DATA = Tuple[List[NODE_NAME], List[LOGIC_FUNC]]

STATE = np.ndarray

REWARD = int
TERMINATED = bool
TRUNCATED = bool
INFO = dict
GYM_STEP_RETURN = Tuple[STATE, REWARD, TERMINATED, TRUNCATED, INFO]
