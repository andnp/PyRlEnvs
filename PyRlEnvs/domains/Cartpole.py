import numpy as np
from numba import njit

from PyRlEnvs.utils.RandomVariables import DeterministicRandomVariable
from PyRlEnvs.BaseEnvironment import BaseEnvironment
from PyRlEnvs.utils.numerical import euler

@njit(cache=True)
def _dsdt(sa: np.ndarray, t: float):
    g = 9.8
    l = 0.5
    masspole = 0.1
    masscart = 1.0

    polemass_length = masspole * l
    total_mass = masspole + masscart

    x, dx, theta, dtheta, force = sa

    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)

    term1: float = (force + polemass_length * dtheta**2 * sinTheta) / total_mass
    ddtheta: float = (g * sinTheta - cosTheta * term1) / (l * (4 / 3 - masspole * cosTheta**2 / total_mass))
    ddx: float = term1 - polemass_length * ddtheta * cosTheta / total_mass

    return np.array([dx, ddx, dtheta, ddtheta, 0])

@njit(cache=True)
def _isTerminal(s: np.ndarray) -> bool:
    x, _, theta, _ = s
    theta_thresh = 12 * 2 * np.pi / 360
    return x < -2.4 or x > 2.4 or theta < -theta_thresh or theta > theta_thresh

def _nextState(s: np.ndarray, a: int, dt: float) -> np.ndarray:
    force = 10 if a == 1 else -10

    sa = np.append(s, force)
    spa = euler(_dsdt, sa, np.array([0, dt]))

    # only need the last result of the integration
    spa = spa[-1]
    sp = spa[:-1]

    return sp

class Cartpole(BaseEnvironment):
    dt = 0.02

    @classmethod
    def nextStates(cls, s: np.ndarray, a: int):
        sp = _nextState(s, a, cls.dt)
        return DeterministicRandomVariable(sp)

    @classmethod
    def actions(cls, s: np.ndarray):
        return [0, 1]

    @classmethod
    def reward(cls, s: np.ndarray, a: int, sp: np.ndarray):
        return 1.0

    @classmethod
    def terminal(cls, s: np.ndarray, a: int, sp: np.ndarray):
        return _isTerminal(sp)

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._state = np.zeros(4)

    def start(self):
        start = self.rng.uniform(-0.05, 0.05, size=4)
        self._start = start
        return start

    def step(self, action: int):
        sp = self.nextStates(self._state, action).sample(self.rng)
        r = self.reward(self._state, action, sp)
        t = self.terminal(self._start, action, sp)

        self._state = sp

        return (r, sp, t)

    def setState(self, state: np.ndarray):
        self._start = state.copy()

    def copy(self, seed: int):
        m = CartPole(seed)
        m._state = self._state.copy()
        return m
