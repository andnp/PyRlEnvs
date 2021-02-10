import numpy as np
from numba import njit
from PyRlEnvs.BaseEnvironment import BaseEnvironment

@njit(cache=True)
def _nextState(s, a):
    a = a - 1
    p, v = s

    v += 0.001 * a - 0.0025 * np.cos(3 * p)

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
    def nextStates(s, a):
        return [_nextState(s, a)]

    @staticmethod
    def actions(s):
        return [0, 1, 2]

    @staticmethod
    def reward(s, a, sp):
        return -1

    @staticmethod
    def terminal(s, a, sp):
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

    def step(self, a):
        # deterministic next state, so no need to sample
        sp = MountainCar.nextStates(self._state, a)[0]
        r = MountainCar.reward(self._state, a, sp)
        t = MountainCar.terminal(self._state, a, sp)

        self._state = sp

        return (r, sp.copy(), t)
