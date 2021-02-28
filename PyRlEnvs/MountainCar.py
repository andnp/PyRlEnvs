import numpy as np
from numba import njit
from PyRlEnvs.BaseEnvironment import BaseEnvironment

@njit(cache=True)
def _nextState(s: np.ndarray, a: int):
    a = a - 1
    p: float = s[0]
    v: float = s[1]

    v = v + 0.001 * a - 0.0025 * np.cos(3 * p)

    if v < -0.07:
        v = -0.07
    elif v >= 0.07:
        v = 0.07

    p += v

    if p >= 0.5:
        return np.array([p, v])

    if p < -1.2:
        return np.array([-1.2, 0.0])

    return np.array([p, v])

class MountainCar(BaseEnvironment):
    @staticmethod
    def nextStates(s: np.ndarray, a: int):
        return [_nextState(s, a)]

    @staticmethod
    def actions(s: np.ndarray):
        return [0, 1, 2]

    @staticmethod
    def reward(s: np.ndarray, a: int, sp: np.ndarray):
        return -1

    @staticmethod
    def terminal(s: np.ndarray, a: int, sp: np.ndarray):
        p, _ = sp

        return p >= 0.5

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        self._state = np.array([0, 0])

    def start(self):
        position = -0.6 + self.rng.random() * 0.2
        velocity = 0

        start = np.array([position, velocity])
        self._state = start

        return start

    def step(self, action: int):
        # deterministic next state, so no need to sample
        sp = MountainCar.nextStates(self._state, action)[0]
        r = MountainCar.reward(self._state, action, sp)
        t = MountainCar.terminal(self._state, action, sp)

        self._state = sp

        return (r, sp.copy(), t)

    def setState(self, state: np.ndarray):
        self._state = state.copy()

    def copy(self, seed: int):
        m = MountainCar(seed)
        m._state = self._state.copy()
        return m
