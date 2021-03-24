import numpy as np
from numba import njit

@njit(cache=True)
def wrap(x: float, mi: float, ma: float):
    d = ma - mi

    while x > ma:
        x = x - d

    while x < mi:
        x = x + d

    return x

@njit(cache=True)
def clipEach(x: np.ndarray, mi: float, ma: float):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = clip(x[i], mi, ma)

    return out

@njit(cache=True)
def clip(x: float, mi: float, ma: float):
    if x > ma:
        return ma

    if x < mi:
        return mi

    return x

def immutable(arr: np.ndarray):
    arr.setflags(write=False)
    return arr
