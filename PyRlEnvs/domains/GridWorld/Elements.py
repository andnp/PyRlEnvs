import numpy as np
from abc import abstractmethod
from .utils import Coords, getState, predecessors

class Element:
    def __init__(self, name: str = ''):
        self.name = name
        self.shape: Coords = (0, 0)

    # deferred initialization for when we know more about our gridworld
    def init(self, shape: Coords):
        self.shape = shape

    def getState(self, coords: Coords) -> int:
        return getState(coords, self.shape)

    # Used for printing to console. Can be safely ignored without impacting behavior
    def trigger(self, s: int, a: int, sp: int) -> bool:
        return False

    @abstractmethod
    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        pass

class GoalState(Element):
    def __init__(self, loc: Coords, reward: float):
        super().__init__('Goal')
        self.loc = loc
        self.reward = reward

    def init(self, shape: Coords):
        self.shape = shape

        self.sp = self.getState(self.loc)

    def trigger(self, s: int, a: int, sp: int):
        return sp == self.sp

    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        for (s, a) in predecessors(self.sp, self.shape):
            R[s, a, self.sp] = self.reward
            T[s, a, self.sp] = 1

class StartState(Element):
    def __init__(self, loc: Coords, weight: float = 1):
        super().__init__('Start')
        self.loc = loc
        self.weight = weight

    def init(self, shape: Coords):
        self.shape = shape

        self.s = self.getState(self.loc)

    def trigger(self, s: int, a: int, sp: int):
        return s == self.s

    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        d0[self.s] = self.weight

class WallState(Element):
    def __init__(self, loc: Coords):
        super().__init__('Wall')
        self.loc = loc

    def init(self, shape: Coords):
        self.shape = shape

        self.sp = self.getState(self.loc)

    def trigger(self, s: int, a: int, sp: int):
        return sp == self.sp

    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        sp = self.sp

        for (s, a) in predecessors(sp, self.shape):
            d0[sp] = 0
            K[s, a, s] = 1
            R[s, a, s] = R[s, a, sp]
            K[:, :, sp] = 0
            R[:, :, sp] = 0
