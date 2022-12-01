from __future__ import annotations

import numpy as np
from abc import abstractmethod
from typing import Dict, Optional, Union

RNG = np.random.Generator

class Distribution:
    def __init__(self, rng: Optional[RNG] = None):
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def sample(self, rng: Optional[RNG] = None) -> float:
        pass

    def _rng(self, rng: Optional[RNG] = None):
        if rng:
            return rng

        if self.rng:
            return self.rng

        return np.random

    def __add__(self, other: Union[float, Distribution]):
        if isinstance(other, float):
            other = DeltaDist(other, self.rng)

        return AddedDist(self, other)

    def __mul__(self, other: Union[float, Distribution]):
        if isinstance(other, float):
            other = DeltaDist(other, self.rng)

        return MultipliedDist(self, other)

    def __rmul__(self, other: Union[float, Distribution]):
        return self.__mul__(other)

class AddedDist(Distribution):
    def __init__(self, ldist: Distribution, rdist: Distribution):
        self.ldist = ldist
        self.rdist = rdist

    def sample(self, rng: Optional[RNG] = None):
        return self.ldist.sample(rng) + self.rdist.sample(rng)

class MultipliedDist(Distribution):
    def __init__(self, ldist: Distribution, rdist: Distribution):
        self.ldist = ldist
        self.rdist = rdist

    def sample(self, rng: Optional[RNG] = None):
        return self.ldist.sample(rng) * self.rdist.sample(rng)

class DeltaDist(Distribution):
    def __init__(self, val: float, rng: Optional[RNG] = None):
        super().__init__(rng)

        self.value = val

    def sample(self, rng: Optional[RNG] = None) -> float:
        return self.value

class Gaussian(Distribution):
    def __init__(self, mean: float, stddev: float, rng: Optional[RNG] = None):
        super().__init__(rng)

        self.mean = mean
        self.stddev = stddev

    def sample(self, rng: Optional[RNG] = None):
        return self._rng(rng).normal(self.mean, self.stddev)

class Uniform(Distribution):
    def __init__(self, mi: float, ma: float, rng: Optional[RNG] = None):
        super().__init__(rng)

        self.mi = mi
        self.ma = ma

    def sample(self, rng: Optional[RNG] = None):
        return self._rng(rng).uniform(self.mi, self.ma)


class Gamma(Distribution):
    def __init__(self, shape: float, scale: float, rng: Optional[RNG] = None):
        super().__init__(rng)

        self.shape = shape
        self.scale = scale

    def sample(self, rng: Optional[RNG] = None):
        return self._rng(rng).gamma(self.shape, self.scale)

# TODO: well this is wrong... Should use rejection sampling
class ClippedGaussian(Gaussian):
    def __init__(self, mean: float, stddev: float, mi: float, ma: Optional[float] = None, rng: Optional[RNG] = None):
        super().__init__(mean, stddev, rng)

        self.mi = mi
        self.ma = ma

    def sample(self, rng: Optional[RNG] = None):
        x = super().sample(rng)
        return np.clip(x, self.mi, self.ma)

def sampleChildren(d: Dict[str, Distribution], rng: Optional[RNG] = None):
    out = {}
    for k in d:
        out[k] = d[k].sample(rng)

    return out
