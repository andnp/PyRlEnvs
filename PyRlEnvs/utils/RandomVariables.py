import numpy as np
from abc import abstractmethod
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union, cast
from PyRlEnvs.utils.random import NpArray, sample

T = TypeVar('T')

FloatArray = Union[Sequence[float], np.ndarray]
RNG = np.random.Generator

class RandomVariable(Generic[T]):
    def __init__(self, rng: Optional[RNG] = None):
        if rng is None:
            rng = np.random.default_rng()

        self.rng: RNG = rng

    def _rng(self, rng: Optional[RNG] = None) -> RNG:
        if rng is not None:
            return rng

        return self.rng

    @abstractmethod
    def sample(self, rng: Optional[RNG] = None) -> T:
        pass

class DeterministicRandomVariable(RandomVariable[T]):
    def __init__(self, val: T, rng: Optional[RNG] = None):
        super().__init__(rng)

        self.val = val

    def sample(self, rng: Optional[RNG] = None):
        return self.val

    def __iter__(self):
        yield self.val

class DiscreteRandomVariable(RandomVariable[T]):
    def __init__(self, vals: Sequence[T], probs: FloatArray, rng: Optional[RNG] = None):
        super().__init__(rng)

        self.vals = vals
        self.probs = probs

    def sample(self, rng: Optional[RNG] = None) -> T:
        rng = self._rng(rng)
        idx = sample(self.probs, rng)
        return self.vals[idx]

    def __iter__(self):
        return self.vals.__iter__()

    @staticmethod
    def fromProbs(probs: NpArray, rng: Optional[RNG] = None):
        vals = cast(Sequence[int], np.arange(len(probs)))
        return DiscreteRandomVariable[int](vals, probs, rng)

    @staticmethod
    def fromUniform(vals: Sequence[T], rng: Optional[RNG] = None):
        probs: np.ndarray = np.ones(len(vals)) / len(vals)
        return DiscreteRandomVariable[T](vals, probs, rng)

class InferedRandomVariable(RandomVariable[T]):
    def __init__(self, sampling_func: Callable[..., T], rng: Optional[RNG] = None):
        super().__init__(rng)

        self.sampling_func = sampling_func

    def sample(self, *args: Any, rng: Optional[RNG] = None, **kwargs: Any):
        rng = self._rng(rng)
        return self.sampling_func(rng, *args, **kwargs)
