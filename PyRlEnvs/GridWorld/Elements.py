import numpy as np
from abc import abstractmethod
from .utils import Coords, getState

class Element:
    def __init__(self, name:str = ''):
        self.name = name
        self.shape: Coords = (0, 0)

    # deferred initialization for when we know more about our gridworld
    def init(self, shape: Coords):
        self.shape = shape

    def getState(self, coords: Coords) -> int:
        return getState(coords, self.shape)

    @abstractmethod
    def trigger(self, s: int, a: int, sp: int): ...

    @abstractmethod
    def apply(self, s: int, a: int, sp: int, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray): ...

class GoalState(Element):
    def __init__(self, loc: Coords, reward: float):
        super().__init__('Goal')
        self.loc = loc
        self.reward = reward

    def init(self, shape: Coords):
        self.shape = shape

        self.sp = self.getState(self.loc)

    def trigger(self, s:int, a:int, sp:int):
        return sp == self.sp

    def apply(self, s: int, a: int, sp: int, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        R[s, a, sp] = self.reward
        T[s, a, sp] = 1

class StartState(Element):
    def __init__(self, loc: Coords, weight: float = 1):
        super().__init__('Start')
        self.loc = loc
        self.weight = weight

    def init(self, shape: Coords):
        self.shape = shape

        self.s = self.getState(self.loc)

    def trigger(self, s:int, a:int, sp:int):
        return s == self.s

    def apply(self, s: int, a: int, sp: int, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        d0[s] = self.weight

class WallState(Element):
    def __init__(self, loc: Coords):
        super().__init__('Wall')
        self.loc = loc

    def init(self, shape: Coords):
        self.shape = shape

        self.sp = self.getState(self.loc)

    def trigger(self, s:int, a:int, sp:int):
        return sp == self.sp

    def apply(self, s: int, a: int, sp: int, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        K[s, a, s] = 1
        R[s, a, s] = R[s, a, sp]
        K[s, a, sp] = 0
        R[s, a, sp] = 0
