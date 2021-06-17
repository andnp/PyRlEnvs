import numpy as np
from abc import abstractmethod
from typing import Any
import RlGlue


class BaseEnvironment(RlGlue.BaseEnvironment):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def setState(self, state: Any):
        pass

    @abstractmethod
    def copy(self, seed: int = 0) -> RlGlue.BaseEnvironment:
        pass
