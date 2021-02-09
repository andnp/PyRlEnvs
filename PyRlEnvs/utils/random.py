from typing import Any, Sequence, TypeVar, Union
import numpy as np
from numba import njit

T = TypeVar('T')
AnyNumber = Union[float, int]
NpList = Union[np.ndarray, Sequence[AnyNumber]]

# we can speed this up if we abstract away from the rng
@njit(cache=True)
def _sample(arr, r):
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we should never get here
    return len(arr) - 1

# way faster than np.random.choice
# arr is an array of probabilities, should sum to 1
def sample(arr: NpList, rng: Any = np.random):
    # if we can avoid incrementing the rng, do so
    if len(arr) == 1:
        return arr[0]

    r = rng.random()
    return _sample(arr, r)

# also much faster than np.random.choice
# choose an element from a list with uniform random probability
def choice(arr: Sequence[T], rng: Any = np.random) -> T:
    idxs = rng.permutation(len(arr))
    return arr[idxs[0]]