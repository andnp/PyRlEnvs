from abc import abstractmethod
from typing import Any
import RlGlue


class BaseEnvironment(RlGlue.BaseEnvironment):
    @abstractmethod
    def setState(self, state: Any):
        pass

    @abstractmethod
    def copy(self, seed: int = 0) -> RlGlue.BaseEnvironment:
        pass
