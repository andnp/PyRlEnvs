import numpy as np
from functools import partial

from PyRlEnvs.Category import addToCategory
from PyRlEnvs.utils.distributions import ClippedGaussian, DeltaDist, Gamma, Gaussian, Uniform, sampleChildren
from PyRlEnvs.BaseEnvironment import BaseEnvironment
from PyRlEnvs.utils.math import try2jit
from PyRlEnvs.utils.numerical import euler

@try2jit
def _dsdt(g: float, l: float, masspole: float, masscart: float, sa: np.ndarray, t: float):
    polemass_length = masspole * l
    total_mass = masspole + masscart

    x, dx, theta, dtheta, force = sa

    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)

    term1: float = (force + polemass_length * dtheta**2 * sinTheta) / total_mass
    ddtheta: float = (g * sinTheta - cosTheta * term1) / (l * (4 / 3 - masspole * cosTheta**2 / total_mass))
    ddx: float = term1 - polemass_length * ddtheta * cosTheta / total_mass

    return np.array([dx, ddx, dtheta, ddtheta, 0])

@try2jit
def _isTerminal(s: np.ndarray) -> bool:
    x, _, theta, _ = s
    theta_thresh = 12 * 2 * np.pi / 360
    return x < -2.4 or x > 2.4 or theta < -theta_thresh or theta > theta_thresh

class Cartpole(BaseEnvironment):
    physical_constants = {
        # physics of world / cart / pole
        'gravity': 9.8,
        'pole_length': 0.5,
        'pole_mass': 0.1,
        'cart_mass': 1.0,
    }

    per_step_constants = {
        # for integration fidelity
        'dt': DeltaDist(0.02),

        # controller force
        'force': DeltaDist(10),
    }

    randomized_constants = {
        'gravity': ClippedGaussian(mean=9.8, stddev=2.0, mi=5.0, ma=13.8),
        'pole_length': Uniform(mi=0.3, ma=0.7),
        'pole_mass': Uniform(mi=0.05, ma=0.15),
        'cart_mass': Uniform(mi=0.6, ma=1.4),
    }

    per_step_random_constants = {
        # make time sampled from a normal distribution (clipped to ensure non-negative) with a long-tailed nuisance
        # distribution to simulate random delays or interference
        'dt': 0.95 * ClippedGaussian(mean=0.02, stddev=0.01, mi=0.015, ma=None) + 0.05 * Gamma(shape=0.01, scale=2.0),

        'force': Gaussian(mean=10, stddev=1.0),
    }

    def __init__(self, randomize: bool = False, seed: int = 0):
        super().__init__(seed)
        self.randomize = randomize
        self._state = np.zeros(4)

        self.start_rng = np.random.default_rng(seed)

        if randomize:
            self.physical_constants = sampleChildren(self.randomized_constants, self.rng)
            self.per_step_constants = self.per_step_random_constants

        self._dsdt = partial(_dsdt,
            self.physical_constants['gravity'],
            self.physical_constants['pole_length'],
            self.physical_constants['pole_mass'],
            self.physical_constants['cart_mass'],
        )

    # -------------------------
    # -- Dynamics equations --
    # -------------------------

    def nextState(self, s: np.ndarray, a: int):
        # get per-step constants
        dt = self.per_step_constants['dt'].sample(self.rng)
        force = self.per_step_constants['force'].sample(self.rng)

        force = force if a == 1 else -force

        sa = np.append(s, force)
        spa = euler(self._dsdt, sa, np.array([0, dt]))

        # only need the last result of the integration
        spa = spa[-1]
        sp = spa[:-1]

        return sp

    def actions(self, s: np.ndarray):
        return [0, 1]

    def reward(self, s: np.ndarray, a: int, sp: np.ndarray):
        return 1.0

    def terminal(self, s: np.ndarray, a: int, sp: np.ndarray):
        return _isTerminal(sp)

    # ------------------------
    # -- Stateful functions --
    # ------------------------

    def start(self):
        start = self.start_rng.uniform(-0.05, 0.05, size=4)
        self._state = start
        return start

    def step(self, action: int):
        sp = self.nextState(self._state, action)
        r = self.reward(self._state, action, sp)
        t = self.terminal(self._state, action, sp)

        self._state = sp

        return (r, sp, t)

    def setState(self, state: np.ndarray):
        self._state = state.copy()

    def copy(self):
        m = Cartpole(randomize=self.randomize, seed=self._seed)
        m._state = self._state.copy()
        m.physical_constants = self.physical_constants
        m.per_step_constants = self.per_step_constants

        # because we are changing the physics _after_ the derivatives are being precomputed
        # we need to replace the derivatives with the correct eqns for these constants
        m._dsdt = self._dsdt

        return m

class StochasticCartpole(Cartpole):
    def __init__(self, seed: int = 0):
        super().__init__(randomize=True, seed=seed)

class ContinuousActionCartpole(Cartpole):
    def nextState(self, s: np.ndarray, force: float):
        # get per-step constants
        dt = self.per_step_constants['dt'].sample(self.rng)

        force = np.clip(force, -12, 12)
        sa = np.append(s, force)
        spa = euler(self._dsdt, sa, np.array([0, dt]))

        # only need the last result of the integration
        spa = spa[-1]
        sp = spa[:-1]

        return sp

addToCategory('classic-control', Cartpole)
addToCategory('stochastic', StochasticCartpole)
