import numpy as np


def booleanize(X, l):
    """Turn an int to a boolean string of length l"""
    output = np.zeros((l), dtype=bool)
    for i in range(l):
        h = 2 ** (l - i - 1)
        if X >= h:
            X -= h
            output[i] = 1
    return output
