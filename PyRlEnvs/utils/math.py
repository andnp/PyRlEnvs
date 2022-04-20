import logging
import numpy as np
from typing import Any, Callable, TypeVar

# -----------------
# -- Numba stuff --
# -----------------

_useFastMath = False
_haveWarned = False
def useFastMath(warn: bool = True):
    global _useFastMath, _haveWarned
    _useFastMath = True

    if not _haveWarned and warn:
        _haveWarned = True
        logging.getLogger('PyRlEnvs').warning('Enabling fastmath can result in nondeterministic accumulation of floating point errors')

F = TypeVar('F', bound=Callable[..., Any])
def try2jit(f: F) -> F:
    try:
        from numba import njit
        return njit(f, cache=True, nogil=True, fastmath=_useFastMath)

    except Exception:
        return f

# many of our functions operate on scalars or have operations which do not
# accumulate floating point errors. These can safely use fastmath
# without incurring nondeterminism.
def try2FastJit(f: F) -> F:
    try:
        from numba import njit
        return njit(f, cache=True, nogil=True, fastmath=True)

    except Exception:
        return f

# ----------------
# -- Math stuff --
# ----------------

@try2FastJit
def wrap(x: float, mi: float, ma: float):
    d = ma - mi

    while x > ma:
        x = x - d

    while x < mi:
        x = x + d

    return x

@try2FastJit
def clipEach(x: np.ndarray, mi: float, ma: float):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = clip(x[i], mi, ma)

    return out

@try2FastJit
def clip(x: float, mi: float, ma: float):
    if x > ma:
        return ma

    if x < mi:
        return mi

    return x

@try2FastJit
def clipInt(x: int, mi: int, ma: int):
    if x >= ma:
        return ma - 1

    if x < mi:
        return mi

    return x

def immutable(arr: np.ndarray):
    arr.setflags(write=False)
    return arr
