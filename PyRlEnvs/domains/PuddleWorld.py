from functools import partial
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from PyRlEnvs.BaseEnvironment import BaseEnvironment

from numba import njit

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

@njit(cache=True)
def l2_dist(x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.power(x - y, 2).sum())

@njit(cache=True)
def puddleDist(
        # parameters of the puddle
        head: np.ndarray,
        tail: np.ndarray,
        radius: float,
        length: float,
        axis: int,

        # agent parameters
        pos: np.ndarray,
) -> float:
    u = (pos - tail)[axis] / length

    dist = 0.0

    if u < 0.0:
        dist = l2_dist(tail, pos)
    elif u > 1.0:
        dist = l2_dist(head, pos)
    else:
        pos_ = tail + u * (head - tail)
        dist = l2_dist(pos, pos_)

    if dist < radius:
        return (radius - dist)

    return 0.

def buildPuddle(
        head: npt.ArrayLike,
        tail: npt.ArrayLike,
        radius: float,
        length: float,
        axis: int,
):

    head_ = np.asarray(head)
    tail_ = np.asarray(tail)

    return partial(puddleDist, head_, tail_, radius, length, axis)

class PuddleWorld(BaseEnvironment):
    def __init__(self, seed: int = 0):
        super().__init__(seed)

        # two rngs. One gives a sequence of perturbations per step
        # the other gives a sequence of start states
        self.rng = default_rng(seed)
        self.start_rng = default_rng(seed)

        self.state = np.zeros(2)

        # Add the puddles
        self.puddle1 = buildPuddle(
            head=(0.45, 0.75),
            tail=(0.1, 0.75),
            radius=0.1,
            length=0.35,
            axis=0,
        )

        self.puddle2 = buildPuddle(
            head=(0.45, 0.8),
            tail=(0.45, 0.4),
            radius=0.1,
            length=0.4,
            axis=1,
        )

        self.goal_dimension = 0.05
        self.def_displacement = 0.05

        self.sigma = 0.01

        self.goal = 1 - self.goal_dimension

    @classmethod
    def actions(cls, s):
        return [UP, DOWN, RIGHT, LEFT]

    def start(self):
        # start in a region around (0.05, 0.05) randomly
        # enforce boundaries of 1x1 square
        self.state = self.start_rng.normal(0.05, 0.1, size=2)
        self.state = np.clip(self.state, 0, 1)
        return np.copy(self.state)

    def _terminal(self):
        return np.all(self.state >= self.goal)

    def _reward(self, state, terminal):
        if terminal:
            return -1.

        reward = -1 + self.puddle1(state) * -400 + self.puddle2(state) * -400
        return reward

    def step(self, a):
        s = self.state

        n = self.rng.normal(scale=self.sigma)

        if a == UP: # up
            s[1] += (self.def_displacement + n)
        elif a == DOWN: # down
            s[1] -= (self.def_displacement + n)
        elif a == RIGHT: # right
            s[0] += (self.def_displacement + n)
        elif a == LEFT: # left
            s[0] -= (self.def_displacement + n)
        else:
            raise Exception()

        if s[0] > 1:
            s[0] = 1
        elif s[0] < 0:
            s[0] = 0

        if s[1] > 1:
            s[1] = 1
        elif s[1] < 0:
            s[1] = 0

        self.state = s

        t = self._terminal()
        r = self._reward(s, t)
        gamma = 0.0 if t else 1.0

        return (r, np.copy(self.state), t, {'gamma': gamma})
