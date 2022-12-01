"""doc
TODO: find original citation
"""

import numpy as np
from functools import partial
from PyRlEnvs.Category import addToCategory
from PyRlEnvs.utils.math import clip, try2jit, wrap
from PyRlEnvs.utils.numerical import rungeKutta
from PyRlEnvs.BaseEnvironment import BaseEnvironment
from PyRlEnvs.utils.random import makeSampler, sampleDict, truncatedGaussian, uniform, gamma, gaussian

@try2jit
def _dsdt(l1: float, m1: float, m2: float, com1: float, com2: float, g: float, sa: np.ndarray, t: float):
    moi = 1.    # moment of inertia

    a = sa[-1]
    theta1, theta2, dtheta1, dtheta2 = sa[:-1]

    d1: float = m1 * com1**2 + m2 * (l1**2 + com2**2 + 2 * l1 * com2 * np.cos(theta2)) + 2 * moi
    d2: float = m2 * (com2**2 + l1 * com2 * np.cos(theta2)) + moi

    phi2: float = m2 * com2 * g * np.cos(theta1 + theta2 - (np.pi / 2))
    phi1: float = -m2 * l1 * com2 * dtheta2**2 * np.sin(theta2) - 2 * m2 * l1 * com2 * dtheta2 * dtheta1 * np.sin(theta2) + (m1 * com1 + m2 * l1) * g * np.cos(theta1 - (np.pi / 2)) + phi2

    ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * com2 * dtheta1**2 * np.sin(theta2) - phi2) / (m2 * com2**2 + moi - (d2**2 / d1))
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.])

@try2jit
def _transform(s: np.ndarray) -> np.ndarray:
    theta1, theta2, dtheta1, dtheta2 = s

    return np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), dtheta1, dtheta2])

@try2jit
def _isTerminal(s: np.ndarray) -> bool:
    return -np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.

class Acrobot(BaseEnvironment):
    # -------------
    # -- Physics --
    # -------------
    physical_constants = {
        'gravity': 9.8,
        'link1_length': 1.,
        'link1_mass': 1.,
        'link1_com': 0.5,  # center of mass
        # link2 length is fixed
        'link2_mass': 1.,
        'link2_com': 0.5
    }

    per_step_constants = {
        'dt': 0.2,
        'force': 1.0,
    }

    randomized_constants = {
        'gravity': makeSampler(lambda: truncatedGaussian(mean=9.8, stddev=2.0, mi=5.0, ma=13.8)),
        'link1_length': makeSampler(lambda: uniform(0.75, 1.25)),
        'link1_mass': makeSampler(lambda: uniform(0.75, 1.25)),
        'link1_com': makeSampler(lambda: truncatedGaussian(mean=0.5, stddev=0.1, mi=0.3, ma=0.7)),
        # link2 length is fixed
        'link2_mass': makeSampler(lambda: uniform(0.75, 1.25)),
        'link2_com': makeSampler(lambda: truncatedGaussian(mean=0.5, stddev=0.1, mi=0.3, ma=0.7)),
    }

    per_step_random_constants = {
        # use clipped gaussian to enforce a lower-bound constraint on how fast we can sample
        # realistically, we can see very long delays but we can never sample faster than the equipment allows
        'dt': makeSampler(lambda: 0.99 * truncatedGaussian(mean=0.2, stddev=0.02, mi=0.125) + 0.01 * gamma(shape=0.1, scale=2.0)),

        # note this isn't clipped, force can flip signs with low probability
        'force': makeSampler(lambda: gaussian(1.0, 0.5)),
    }

    def __init__(self, random_init: bool = False, seed: int = 0):
        super().__init__(seed)
        self.random_init = random_init
        self._state = np.zeros(4)

        self.start_rng = np.random.default_rng(seed)

        if random_init:
            self.physical_constants = sampleDict(self.randomized_constants, self.rng)

        self._dsdt = partial(_dsdt,
            self.physical_constants['link1_length'],
            self.physical_constants['link1_mass'],
            self.physical_constants['link2_mass'],
            self.physical_constants['link1_com'],
            self.physical_constants['link2_com'],
            self.physical_constants['gravity'],
        )

    # -------------------------
    # -- Dynamics equations --
    # -------------------------

    def nextState(self, s: np.ndarray, a: float):
        dt = self.per_step_constants['dt']
        force = self.per_step_constants['force']

        a = (a - 1.) * force

        sa = np.append(s, a)
        spa = rungeKutta(self._dsdt, sa, np.array([0, dt]))

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

    def actions(self, s: np.ndarray):
        return [0, 1, 2]

    def reward(self, s: np.ndarray, a: float, sp: np.ndarray):
        return -1. if not self.terminal(s, a, sp) else 0.

    def terminal(self, s: np.ndarray, a: float, sp: np.ndarray):
        return _isTerminal(sp)

    # ------------------------
    # -- Stateful functions --
    # ------------------------

    def start(self):
        start = self.start_rng.uniform(-.1, .1, size=4)
        self._state = start

        return _transform(start)

    def step(self, action: float):
        sp = self.nextState(self._state, action)
        r = self.reward(self._state, action, sp)
        t = self.terminal(self._state, action, sp)

        self._state = sp

        return (r, _transform(sp), t)

    def setState(self, state: np.ndarray):
        self._state = state.copy()

    def copy(self):
        m = Acrobot(random_init=self.random_init, seed=self._seed)
        m._state = self._state.copy()
        m.physical_constants = self.physical_constants
        m.per_step_constants = self.per_step_constants

        # copy derivative function because state variables changed
        m._dsdt = self._dsdt

        return m

class StochasticAcrobot(Acrobot):
    def __init__(self, random_init: bool = True, seed: int = 0):
        super().__init__(random_init, seed=seed)

addToCategory('classic-control', Acrobot)
addToCategory('stochastic', StochasticAcrobot)
