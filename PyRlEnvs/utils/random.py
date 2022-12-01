from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set, TypeVar, Union
import numpy as np
from numba import njit

from PyRlEnvs.utils.math import clip

T = TypeVar('T')
NpArray = Union[Sequence[float], np.ndarray]
RNG = np.random.Generator

# we can speed this up if we abstract away from the rng
@njit(cache=True)
def _sample(arr: NpArray, r: float) -> int:
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
def sample(arr: NpArray, rng: Any = np.random) -> int:
    # if we can avoid incrementing the rng, do so
    if len(arr) == 1:
        return 0

    r = rng.random()
    return _sample(arr, r)

# also much faster than np.random.choice
# choose an element from a list with uniform random probability
def choice(arr: Union[Sequence[T], Set[T]], rng: Optional[RNG] = None) -> T:
    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, len(arr))
    return list(arr)[idx]


def makeSampler(f: Callable[[], float]):
    @njit
    def _inner(seed: int):
        np.random.seed(seed)
        return f()

    m = np.iinfo(np.int64).max
    def __inner(rng: RNG):
        seed = rng.integers(0, m)
        return _inner(seed)

    return __inner

@njit(cache=True)
def truncatedGaussian(mean: float, stddev: float, mi: Optional[float] = None, ma: Optional[float] = None):
    _tries = 100

    if mi is None:
        mi = -np.inf

    if ma is None:
        ma = np.inf

    x = np.random.normal(mean, stddev)
    while _tries > 0 and (x < mi or x > ma):
        x = np.random.normal(mean, stddev)

    return clip(x, mi, ma)

F = Callable[[RNG], float]
def sampleDict(d: Mapping[str, Union[F, float]], rng: RNG):
    out: Dict[str, float] = {}
    for key in d:
        el = d[key]
        if isinstance(el, float):
            out[key] = el
        else:
            out[key] = el(rng)

    return out

gamma = np.random.gamma
uniform = np.random.uniform
gaussian = np.random.normal
