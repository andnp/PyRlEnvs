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
def clip(x: float, mi: float, ma: float):
    if x > ma:
        return ma

    if x < mi:
        return mi

    return x
