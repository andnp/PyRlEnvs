from __future__ import annotations
import numpy as np
import rlglue
from typing import Any


class BaseEnvironment(rlglue.BaseEnvironment):
    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def setState(self, state: Any):
        raise NotImplementedError

    def copy(self) -> BaseEnvironment:
        raise NotImplementedError
