import numpy as np
from PyRlEnvs.FiniteDynamics import FiniteDynamics

LEFT = 0
RIGHT = 1

def _buildTransitionKernel(states):
    K = np.zeros((states, 2, states))

    mid = int(states // 2)
    last = states - 1

    # handle left side
    K[0, LEFT, mid] = 1
    K[0, RIGHT, 1] = 1

    # handle right side
    K[last, LEFT, last-1] = 1
    K[last, RIGHT, mid] = 1

    # handle rest
    for s in range(1, last, 1):
        K[s, LEFT, s-1] = 1
        K[s, RIGHT, s+1] = 1

    return K

def _buildRewardKernel(states):
    Rs = np.zeros((states, 2, states))

    last = states - 1

    Rs[0, LEFT] = -1
    Rs[last, RIGHT] = 1

    return Rs

def _buildTerminationKernel(states):
    T = np.zeros((states, 2, states))

    last = states - 1

    T[0, LEFT] = 1
    T[last, RIGHT] = 1

    return T

def _buildStartStateDist(states):
    d0 = np.zeros(states)
    d0[int(states//2)] = 1

    return d0

def buildRandomWalk(states):
    class RandomWalk(FiniteDynamics):
        K = _buildTransitionKernel(states)
        Rs = _buildRewardKernel(states)
        T = _buildTerminationKernel(states)
        d0 = _buildStartStateDist(states)

        num_states = states
        num_actions = 2

    return RandomWalk

# a default class, just for consistency
RandomWalk = buildRandomWalk(5)

# some utility functions to encode other important parts of the problem spec
# not necessarily environment specific, but this is as good a place as any to store them
def _normRows(m):
    return (m.T / np.linalg.norm(m, axis=1)).T

"""
Feature representations used in
TODO: cite <Sutton et al. 2009>
"""
def invertedFeatures(n):
    I = np.eye(n)
    m = 1 - I
    return _normRows(m)

def dependentFeatures(n):
    nfeats = int(np.floor(n/2) + 1)
    m = np.zeros((n, nfeats))

    idx = 0
    for i in range(nfeats):
        m[idx, 0:i+1] = 1
        idx += 1

    for i in range(nfeats-1, 0, -1):
        m[idx, -i:] = 1
        idx += 1

    return _normRows(m)

def tabularFeatures(n):
    return np.eye(n)
