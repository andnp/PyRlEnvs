"""
TODO:
From <cite Baird>
"""

from PyRlEnvs.utils.math import immutable
from PyRlEnvs.Category import addToCategory
import numpy as np
from PyRlEnvs.FiniteDynamics import FiniteDynamics

OUT = 0
IN = 1


def _buildTransitionKernel():
    K = np.zeros((7, 2, 7))
    K[:, OUT, :6] = 1 / 6
    K[:, IN, 6] = 1

    return K

class BairdCounterexample(FiniteDynamics):
    num_states = 7
    num_actions = 2

    K = immutable(_buildTransitionKernel())
    Rs = immutable(np.zeros((7, 2, 7)))

    T = immutable(np.zeros((7, 2, 7)))
    d0 = immutable(np.array([0, 0, 0, 0, 0, 0, 1.]))

# some utility functions to encode other important parts of the problem spec
# not necessarily environment specific, but this is as good a place as any to store them
def behaviorPolicy(s: int):
    return np.array([6 / 7, 1 / 7])

def targetPolicy(s: int):
    return np.array([0, 1.])

def representationMatrix():
    return np.array([
        [1, 2, 0, 0, 0, 0, 0, 0],
        [1, 0, 2, 0, 0, 0, 0, 0],
        [1, 0, 0, 2, 0, 0, 0, 0],
        [1, 0, 0, 0, 2, 0, 0, 0],
        [1, 0, 0, 0, 0, 2, 0, 0],
        [1, 0, 0, 0, 0, 0, 2, 0],
        [2, 0, 0, 0, 0, 0, 0, 1.],
    ])

def initialWeights():
    return np.array([1, 1, 1, 1, 1, 1, 1, 10.])

addToCategory('ope', BairdCounterexample)
addToCategory('ope-counterexample', BairdCounterexample)
addToCategory('finite-dynamics', BairdCounterexample)
