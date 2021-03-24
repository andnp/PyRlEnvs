"""
From Sutton and Barto 2018
Example 11.4, the MRP that shows that the bellman error is "not learnable"

This implementation is set up as an MDP. We recover the original MRP if the behavior policy is uniform random.
"""

from PyRlEnvs.utils.math import immutable
from PyRlEnvs.Category import addToCategory
import numpy as np
from PyRlEnvs.FiniteDynamics import FiniteDynamics

LEFT = 0
RIGHT = 1

A = 0
B = 1
Bp = 2

def _buildTransitionKernel():
    K = np.zeros((3, 2, 3))
    K[A, LEFT, B] = 1
    K[A, RIGHT, Bp] = 1
    K[B, :, A] = 1
    K[Bp, LEFT, B] = 1
    K[Bp, RIGHT, Bp] = 1

    return K

def _buildRewardKernel():
    R = np.zeros((3, 2, 3))
    R[B, :, A] = 1
    R[Bp, LEFT, B] = -1
    R[Bp, RIGHT, Bp] = -1

    return R

class BECounterexample(FiniteDynamics):
    num_states = 3
    num_actions = 2

    K = immutable(_buildTransitionKernel())
    Rs = immutable(_buildRewardKernel())

    T = immutable(np.zeros((3, 2, 3)))
    d0 = immutable(np.array([1., 0, 0]))

# some utility functions to encode other important parts of the problem spec
# not necessarily environment specific, but this is as good a place as any to store them
def behaviorPolicy(s: int):
    return np.array([0.5, 0.5])

def representationMatrix():
    return np.array([
        [1., 0],
        [0, 1],
        [0, 1],
    ])

addToCategory('ope', BECounterexample)
addToCategory('ope-counterexample', BECounterexample)
addToCategory('finite-dynamics', BECounterexample)
