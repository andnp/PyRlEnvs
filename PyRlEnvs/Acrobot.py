import numpy as np
from numba import njit
from PyRlEnvs.utils.math import clip, wrap
from PyRlEnvs.utils.numerical import rungeKutta
from PyRlEnvs.utils.RandomVariables import DeterministicRandomVariable
from PyRlEnvs.BaseEnvironment import BaseEnvironment

@njit(cache=True)
def _dsdt(sa: np.ndarray, t: float):
    # putting these in the method lets numba treat them as constants
    # on the other hand it prevents us from modifying them in derivative classes
    l1 = 1.  # link 1 length
    m1 = 1.  # link 1 mass
    m2 = 1.  # link 2 mass
    com1 = 0.5  # link 1 center of mass
    com2 = 0.5  # link 2 center of mass
    moi = 1.    # moment of inertia

    g = 9.8  # engineer's gravity

    a = sa[-1]
    theta1, theta2, dtheta1, dtheta2 = sa[:-1]

    d1: float = m1 * com1**2 + m2 * (l1**2 + com2**2 + 2 * l1 * com2 * np.cos(theta2)) + 2 * moi
    d2: float = m2 * (com2**2 + l1 * com2 * np.cos(theta2)) + moi

    phi2: float = m2 * com2 * g * np.cos(theta1 + theta2 - (np.pi / 2))
    phi1: float = -m2 * l1 * com2 * dtheta2**2 * np.sin(theta2) - 2 * m2 * l1 * com2 * dtheta2 * dtheta1 * np.sin(theta2) + (m1 * com1 + m2 * l1) * g * np.cos(theta1 - (np.pi / 2)) + phi2

    ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * com2 * dtheta1**2 * np.sin(theta2) - phi2) / (m2 * com2**2 + moi - (d2**2 / d1))
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.])

def _nextState(s: np.ndarray, a: int, dt: float) -> np.ndarray:
    a = a - 1

    sa = np.append(s, a)
    spa = rungeKutta(_dsdt, sa, np.array([0, dt]))

    # only need the last result of the integration
    spa = spa[-1]
    sp = spa[:-1]

    ma_vel1 = 4 * np.pi
    ma_vel2 = 9 * np.pi

    sp[0] = wrap(sp[0], -np.pi, np.pi)
    sp[1] = wrap(sp[1], -np.pi, np.pi)
    sp[2] = clip(sp[2], -ma_vel1, ma_vel1)
    sp[3] = clip(sp[3], -ma_vel2, ma_vel2)

    return sp

@njit(cache=True)
def _transform(s: np.ndarray) -> np.ndarray:
    theta1, theta2, dtheta1, dtheta2 = s

    return np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), dtheta1, dtheta2])

@njit(cache=True)
def _isTerminal(s: np.ndarray) -> bool:
    return -np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.

class Acrobot(BaseEnvironment):
    dt = 0.2

    @classmethod
    def nextStates(cls, s: np.ndarray, a: int):
        sp = _nextState(s, a, cls.dt)
        return DeterministicRandomVariable(_transform(sp))

    @classmethod
    def actions(cls, s: np.ndarray):
        return [0, 1, 2]

    @classmethod
    def reward(cls, s: np.ndarray, a: int, sp: np.ndarray):
        return -1. if not cls.terminal(s, a, sp) else 0.

    @classmethod
    def terminal(cls, s: np.ndarray, a: int, sp: np.ndarray):
        return _isTerminal(sp)

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        self._state = np.zeros(4)

    def start(self):
        start = self.rng.uniform(-.1, .1, size=4)
        self._state = start

        return _transform(start)

    def step(self, action: int):
        sp = _nextState(self._state, action, self.dt)
        r = Acrobot.reward(self._state, action, sp)
        t = Acrobot.terminal(self._state, action, sp)

        self._state = sp

        return (r, _transform(sp), t)

    def setState(self, state: np.ndarray):
        self._state = state.copy()

    def copy(self, seed: int):
        m = Acrobot(seed)
        m._state = self._state.copy()
        return m

# TODO: figure out a way to connect this to the functional api.
# right now, there doesn't appear to be a clean way to do that
# because we don't know the distribution of the nextStates random variable
class JitterAcrobot(Acrobot):
    def __init__(self, noise: float = 0.02, non_normal: float = 0.01, seed: int = 0):
        super().__init__(seed=seed)

        self.noise = noise
        self.non_normal = non_normal

    def step(self, action: int):
        eps = self.rng.normal(0, self.noise)
        nuisance = self.rng.gamma(0.1, 1) - 0.1  # make this 0 mean

        # add some noise to the system, but make sure dt is always positive.
        # assume there's some fundamental constraint that we can only sample at most 8 times per second
        noise = (1 - self.non_normal) * eps + self.non_normal * nuisance
        dt = max(self.dt + noise, 0.125)

        sp = _nextState(self._state, action, dt)
        r = Acrobot.reward(self._state, action, sp)
        t = Acrobot.terminal(self._state, action, sp)

        self._state = sp

        return (r, _transform(sp), t)
