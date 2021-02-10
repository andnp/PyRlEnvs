"""
From Sutton and Barto 2018
Example 11.4, the MRP that shows that the bellman error is "not learnable"

This implementation is set up as an MDP. We recover the original MRP if the behavior policy is uniform random.
"""

import numpy as np
from PyRlEnvs.FiniteDynamics import FiniteDynamics

LEFT = 0
RIGHT = 1

A = 0
B = 1
Bp = 2

def _buildTransitionKernel():
    K = np.zeros((3, 2, 3))
    K[A, B, LEFT] = 1
    K[A, Bp, RIGHT] = 1
    K[B, A] = 1
    K[Bp, B, LEFT] = 1
    K[Bp, Bp, RIGHT] = 1

    return K

class BECounterexample(FiniteDynamics):
    K = _buildTransitionKernel()
    Rs = np.array([
        [0., 0.],
        [1., 0.],
        [-1, -1],
    ])

    T = np.array([0, 0, 1])
    d0 = np.array([1., 0, 0])
