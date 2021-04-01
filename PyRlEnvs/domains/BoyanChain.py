"""
TODO:
From <cite Boyan>
"""

from PyRlEnvs.utils.math import immutable
from PyRlEnvs.Category import addToCategory
import numpy as np
from PyRlEnvs.FiniteDynamics import FiniteDynamics

RIGHT = 0
SKIP = 1


def _buildTransitionKernel():
    K = np.zeros((13, 2, 13))
    for i in range(11):
        K[i, RIGHT, i + 1] = 1
        K[i, SKIP, i + 2] = 1

    K[11, RIGHT, 12] = 1
    K[12, RIGHT, 0] = 1

    return K

def _buildRewardKernel():
    R = np.zeros((13, 2, 13))
    R[:11, :] = -3
    R[11, RIGHT, 12] = -2

    return R

def _buildTerminationKernel():
    T = np.zeros((13, 2, 13))
    T[12, 0, 0] = 1

    return T

class BoyanChain(FiniteDynamics):
    num_states = 13
    num_actions = 2

    K = immutable(_buildTransitionKernel())
    Rs = immutable(_buildRewardKernel())

    T = immutable(_buildTerminationKernel())
    d0 = immutable(np.array([1.] + [0.] * 12))

# some utility functions to encode other important parts of the problem spec
# not necessarily environment specific, but this is as good a place as any to store them
def behaviorPolicy(s: int):
    if s <= 10:
        return np.array([0.5, 0.5])

    return np.array([1.0, 0])

def representationMatrix():
    return np.array([
        [1,    0,    0,    0   ],  # noqa: E241
        [0.75, 0.25, 0,    0   ],  # noqa: E241
        [0.5,  0.5,  0,    0   ],  # noqa: E241
        [0.25, 0.75, 0,    0   ],  # noqa: E241
        [0,    1,    0,    0   ],  # noqa: E241
        [0,    0.75, 0.25, 0   ],  # noqa: E241
        [0,    0.5,  0.5,  0   ],  # noqa: E241
        [0,    0.25, 0.75, 0   ],  # noqa: E241
        [0,    0,    1,    0   ],  # noqa: E241
        [0,    0,    0.75, 0.25],  # noqa: E241
        [0,    0,    0.5,  0.5 ],  # noqa: E241
        [0,    0,    0.25, 0.75],  # noqa: E241
        [0,    0,    0,    1   ],  # noqa: E241
    ])

addToCategory('ope', BoyanChain)
addToCategory('random-walk', BoyanChain)
addToCategory('finite-dynamics', BoyanChain)
