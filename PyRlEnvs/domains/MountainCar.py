import numpy as np
from functools import partial
from PyRlEnvs.utils.distributions import ClippedGaussian, DeltaDist, Gamma, Gaussian, Uniform, sampleChildren
from PyRlEnvs.utils.math import clip, try2jit
from PyRlEnvs.utils.numerical import euler
from PyRlEnvs.Category import addToCategory
from PyRlEnvs.BaseEnvironment import BaseEnvironment

@try2jit
def _dsdt(g: float, m: float, k: float, sa: np.ndarray, t: float):
    p, v, f = sa

    # compute derivative of velocity (acc)
    dv = -g * np.cos(3 * p) + (f / m) - k * v

    return np.array([v, dv, 0.])

# use the hand-crafted (incorrect) euler update found in OpenAI gym and the book
# note that timesteps don't really make sense here, we increase acceleration
# and see an instantaneous change in velocity *and* position afterwards, should
# only see change in velocity at this time, then change in position at *next* time
# keep this around for consistency though
@try2jit
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

    if p < -1.2:
        p = -1.2
        v = 0

    return np.array([p, v])

class MountainCar(BaseEnvironment):
    # -------------
    # -- Physics --
    # -------------
    physical_constants = {
        'gravity': 0.0025,
        'cart_mass': 10,
        'friction': 0.0,
    }

    per_step_constants = {
        'dt': DeltaDist(0.1),
        'force': DeltaDist(.1),
    }

    randomized_constants = {
        # keep stddev low to prevent overly powerful gravity simulations
        # which can make the problem unsolvable, better to err on easier than impossible
        'gravity': ClippedGaussian(mean=0.0025, stddev=0.001, mi=0.001, ma=0.0035),
        'cart_mass': Uniform(mi=8.5, ma=11.5),
        'friction': Uniform(mi=0.0, ma=0.3),
    }

    per_step_random_constants = {
        # use clipped gaussian to enforce a lower-bound constraint on how fast we can sample
        # realistically, we can see very long delays but we can never sample faster than the equipment allows
        'dt': 0.99 * ClippedGaussian(mean=0.1, stddev=0.02, mi=0.05) + 0.01 * Gamma(shape=0.1, scale=1.0),

        # note this isn't clipped, force can flip signs with low probability
        # we use a fairly high stddev to help ensure solvability
        'force': Gaussian(mean=0.1, stddev=0.02),
    }

    def __init__(self, randomize: bool = False, seed: int = 0):
        super().__init__(seed)
        self.randomize = randomize
        self._state = np.array([0, 0])

        self.start_rng = np.random.default_rng(seed)

        if randomize:
            self.physical_constants = sampleChildren(self.randomized_constants, self.rng)
            self.per_step_constants = self.per_step_random_constants

        self._dsdt = partial(_dsdt,
            self.physical_constants['gravity'],
            self.physical_constants['cart_mass'],
            self.physical_constants['friction'],
        )

    def _integrate(self, s: np.ndarray, force: float):
        dt = self.per_step_constants['dt'].sample(self.rng)

        sa = np.append(s, force)
        spa = euler(self._dsdt, sa, np.array([0, dt]))

        # only need last result of integration
        spa = spa[-1]
        sp = spa[:-1]

        # enforce bounds outside of the integrator
        sp[1] = clip(sp[1], -0.07, 0.07)
        if sp[0] < -1.2:
            sp[0] = -1.2
            sp[1] = 0.

        return sp

    def nextState(self, s: np.ndarray, a: int):
        # convert a discrete action into a continuous force
        force = self.per_step_constants['force'].sample(self.rng) * (a - 1)

        return self._integrate(s, force)

    def actions(self, s: np.ndarray):
        return [0, 1, 2]

    def reward(self, s: np.ndarray, a: int, sp: np.ndarray):
        return -1

    def terminal(self, s: np.ndarray, a: int, sp: np.ndarray):
        p, _ = sp

        return p >= 0.5

    def start(self):
        position = -0.6 + self.start_rng.random() * 0.2
        velocity = 0

        start = np.array([position, velocity])
        self._state = start

        return start

    def step(self, action: int):
        sp = self.nextState(self._state, action)
        r = self.reward(self._state, action, sp)
        t = self.terminal(self._state, action, sp)

        gamma = 0.0 if t else 1.0

        self._state = sp

        return (r, sp.copy(), t, {'gamma': gamma})

    def setState(self, state: np.ndarray):
        self._state = state.copy()

    def copy(self):
        m = MountainCar(randomize=self.randomize, seed=self._seed)
        m._state = self._state.copy()
        m.physical_constants = self.physical_constants
        m.per_step_constants = self.per_step_constants

        m._dsdt = self._dsdt

        return m

class GymMountainCar(MountainCar):
    def __init__(self, seed: int = 0):
        super().__init__(randomize=False, seed=seed)

    def nextState(self, s: np.ndarray, a: int):
        return _nextState(s, a)

class StochasticMountainCar(MountainCar):
    def __init__(self, seed: int = 0):
        super().__init__(randomize=True, seed=seed)

class ContinuousActionMountainCar(MountainCar):
    def nextState(self, s: np.ndarray, a: float):
        force = np.clip(a, -3, 3)

        return self._integrate(s, force)


addToCategory('classic-control', MountainCar)
addToCategory('sutton-barto', GymMountainCar)
addToCategory('stochastic', StochasticMountainCar)
