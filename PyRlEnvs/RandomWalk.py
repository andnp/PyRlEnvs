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
