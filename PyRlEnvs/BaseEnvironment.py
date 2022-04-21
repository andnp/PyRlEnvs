from __future__ import annotations
import numpy as np
import RlGlue
from abc import abstractmethod
from typing import Any


class BaseEnvironment(RlGlue.BaseEnvironment):
    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def setState(self, state: Any):
        pass

    @abstractmethod
    def copy(self) -> BaseEnvironment:
        pass
