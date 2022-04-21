"""
TODO:
From <cite Kolter>
"""

import numpy as np
from typing import Optional
from PyRlEnvs.utils.math import immutable
from PyRlEnvs.Category import addToCategory
from PyRlEnvs.FiniteDynamics import FiniteDynamics

S1 = 0
S2 = 1

def _buildTransition():
    K = np.zeros((2, 2, 2))
    K[S1, 0, S1] = 1
    K[S1, 1, S2] = 1
    K[S2, 1, S2] = 1
    K[S2, 0, S1] = 1

    return K

# there are actually infinitely many reward functions that result in the counterexample
# this one is a fairly literal translation from the paper
def _buildReward():
    Rs = np.zeros((2, 2, 2))
    Rs[S1, 0, S1] = -0.01475
    Rs[S1, 1, S2] = -0.01475
    Rs[S2, 1, S2] = 0.03525
    Rs[S2, 0, S1] = 0.03525

    return Rs

class KolterCounterexample(FiniteDynamics):
    num_states = 2
    num_actions = 2

    K = immutable(_buildTransition())
    Rs = immutable(_buildReward())
    T = immutable(np.zeros((2, 2, 2)))
    d0 = immutable(np.array([1., 0]))

# some utility functions to encode other important parts of the problem spec
# not necessarily environment specific, but this is as good a place as any to store them
def buildProblem(eps: float = 0.01, p: Optional[float] = None):
    # if a probability is not specified, then use the counterexample probability
    if p is None:
        p = (2961 + 45240 * eps + 40400 * eps**2) / (4141 + 84840 * eps + 40400 * eps**2)

    mu = immutable(np.array([p, 1 - p]))
    def behavior(s: int):
        return mu

    pi = immutable(np.array([0.5, 0.5]))
    def target(s: int):
        return pi

    X = immutable(np.array([
        [1],
        [1.05 + eps],
    ]))

    gamma = 0.99

    return behavior, target, X, gamma

addToCategory('ope', KolterCounterexample)
addToCategory('ope-counterexample', KolterCounterexample)
addToCategory('finite-dynamics', KolterCounterexample)
