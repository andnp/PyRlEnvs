"""
Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

The state contains concentrations of 6 different cells:
* T1: non-infected CD4+ T-lymphocytes [cells / ml]
* T1*:    infected CD4+ T-lymphocytes [cells / ml]
* T2: non-infected macrophages [cells / ml]
* T2*:    infected macrophages [cells / ml]
* V: number of free HI viruses [copies / ml]
* E: number of cytotoxic T-lymphocytes [cells / ml]

The therapy consists of 2 drugs
(reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
are activated or not. The action space contains therefore of 4 actions:
* *0*: none active
* *1*: RTI active
* *2*: PI active
* *3*: RTI and PI active

Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
Clinical data based optimal STI strategies for HIV:
A reinforcement learning approach
In Proceedings of the 45th IEEE Conference on Decision and Control (2006).
"""

import numpy as np
from PyRlEnvs.Category import addToCategory
from PyRlEnvs.utils.math import clipEach, try2jit
from PyRlEnvs.utils.RandomVariables import DeterministicRandomVariable
from PyRlEnvs.BaseEnvironment import BaseEnvironment
from scipy.integrate import solve_ivp

@try2jit
def _dsdt(t: float, sa: np.ndarray):
    # putting these in the method lets numba treat them as constants
    # on the other hand it prevents us from modifying them in derivative classes
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = .34
    k1 = 8e-7
    k2 = 1e-4
    delta = .7
    m1 = 1e-5
    m2 = 1e-5
    nt = 100
    c = 13.
    rho1 = 1.
    rho2 = 1.
    lambdaE = 1
    be = 0.3
    kb = 100
    de = 0.25
    kd = 500
    deltaE = 0.1

    t1, t2, t1s, t2s, v, e, eps1, eps2 = sa

    tmp1 = (1. - eps1) * k1 * v * t1
    tmp2 = (1. - f * eps1) * k2 * v * t2

    dt1 = lambda1 - d1 * t1 - tmp1
    dt2 = lambda2 - d2 * t2 - tmp2
    dt1s = tmp1 - delta * t1s - m1 * e * t1s
    dt2s = tmp2 - delta * t2s - m2 * e * t2s

    dv = (1 - eps2) * nt * delta * (t1s - t2s) - c * v - ((1 - eps1) * rho1 * k1 * t1 + (1 - f * eps1) * rho2 * k2 * t2) * v
    de = lambdaE + be * (t1s - t2s) / (t1s + t2s + kb) * e - de * (t1s - t2s) / (t1s + t2s + kd) * e - deltaE * e

    return np.array([dt1, dt2, dt1s, dt2s, dv, de, 0, 0])

def _nextState(s: np.ndarray, a: int, dt: float, effects: np.ndarray) -> np.ndarray:
    eps1, eps2 = effects[a]
    sa = np.append(s, [eps1, eps2])
    spa = solve_ivp(_dsdt, [0, dt], sa).y.T

    # only need the last result of the integration
    spa = spa[-1]
    sp = spa[:-2]

    sp = clipEach(sp, 1e-8, 1e8)

    return sp

@try2jit
def _transform(s: np.ndarray) -> np.ndarray:
    return np.log10(s)

class HIVTreatment(BaseEnvironment):
    dt = 5

    actionEffects = np.array([[0., 0], [.7, 0], [0, .3], [.7, .3]])

    @classmethod
    def nextStates(cls, s: np.ndarray, a: int):
        sp = _nextState(s, a, cls.dt, cls.actionEffects)
        return DeterministicRandomVariable(_transform(sp))

    @classmethod
    def actions(cls, s: np.ndarray):
        return [0, 1, 2, 3]

    @classmethod
    def reward(cls, s: np.ndarray, a: int, sp: np.ndarray):
        _, _, _, _, v, e = sp

        eps1, eps2 = cls.actionEffects[a]
        return -0.1 * v - 2e4 * eps1**2 - 2e3 * eps2**2 + 1e3 * e

    @classmethod
    def terminal(cls, s: np.ndarray, a: int, sp: np.ndarray):
        return False

    def __init__(self, random_start: float = 0, seed: int = 0):
        super().__init__(seed)
        self.random_start = random_start
        self.start_rng = np.random.default_rng(seed)

        self._state: np.ndarray = np.zeros(6)

    def start(self):
        # perturb the start state multiplicatively by a normal random amount for each component of system
        eps = 1
        if self.random_start > 0:
            eps = self.start_rng.normal(1, self.random_start, size=6)

        # start in non-healthy stable state
        start = np.array([163573, 5., 11945, 46, 63919, 24]) * eps
        self._state = start

        return _transform(start)

    def step(self, action: int):
        sp = _nextState(self._state, action, self.dt, self.actionEffects)
        r = HIVTreatment.reward(self._state, action, sp)
        t = HIVTreatment.terminal(self._state, action, sp)

        gamma = 0.0 if t else 1.0

        self._state = sp

        return (r, _transform(sp), t, {'gamma': gamma})

    def setState(self, state: np.ndarray):
        self._state = state.copy()

    def copy(self):
        m = HIVTreatment(self._seed)
        m._state = self._state.copy()
        return m

addToCategory('classic-control', HIVTreatment)
