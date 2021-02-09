from abc import abstractmethod
from typing import Callable
import numpy as np
from numba import njit
from PyRlEnvs.BaseEnvironment import BaseEnvironment
from PyRlEnvs.utils.random import sample

@njit(cache=True)
def _actions(K, s: int):
    return np.unique(np.where(K[s] != 0)[0])

@njit(cache=True)
def _nextStates(K, s: int, a: int):
    return [sp[0] for sp in np.where(K[s, a, :] != 0)]

@njit(cache=True)
def _transitionMatrix(K, T, pi, gamma):
    states = pi.shape[0]
    P = np.zeros((states, states))
    G = 1 - T
    if gamma == 1:
        G = np.ones_like(G)

    for s in range(states):
        for sp in range(states):
            P[s, sp] = np.sum(K[s, :, sp] * pi[s] * G[s, :, sp]) * gamma

    return P

@njit(cache=True)
def _averageReward(K, Rs, pi):
    states = pi.shape[0]

    R = np.zeros(states)
    for s in range(states):
        for sp in range(states):
            R[s] += np.sum(K[s, :, sp] * Rs[s, :, sp] * pi[s])

    return R

@njit(cache=True)
def _stateDistribution(P):
    return np.linalg.matrix_power(P, 1000).mean(axis=0)

class FiniteDynamics(BaseEnvironment):
    # start state dist
    d0 = np.zeros(0)

    # transition kernel
    # shape: (states, actions, states)
    K = np.zeros(0)

    # reward kernel
    # shape: (states, actions, states)
    Rs = np.zeros(0)

    # termination kernel
    # shape: (states, actions, states)
    T = np.zeros(0)

    num_states = 0
    num_actions = 0

    @classmethod
    def actions(cls, s: int):
        # the available actions are any action where the kernel is non-zero for the given state
        return _actions(cls.K, s)

    @classmethod
    def nextStates(cls, s: int, a: int):
        return _nextStates(cls.K, s, a)

    @classmethod
    def reward(cls, s: int, a: int, sp: int):
        return cls.Rs[s, a, sp]

    @classmethod
    def terminal(cls, s: int, a: int, sp: int):
        return bool(cls.T[s, a, sp])

    @classmethod
    def constructTransitionMatrix(cls, policy: Callable[[int], np.ndarray], gamma=None):
        if gamma is None:
            gamma = 1

        states = cls.num_states
        pi = np.array([ policy(s) for s in range(states) ])

        return _transitionMatrix(cls.K, cls.T, pi, gamma)

    @classmethod
    def constructRewardVector(cls, policy: Callable[[int], np.ndarray]):
        states = cls.num_states
        pi = np.array([ policy(s) for s in range(states) ])

        return _averageReward(cls.K, cls.Rs, pi)

    @classmethod
    def computeStateDistribution(cls, policy: Callable[[int], np.ndarray]):
        P = cls.constructTransitionMatrix(policy)
        return _stateDistribution(P)

    def __init__(self, seed: int = 0):
        super().__init__()

        self.rng = np.random.RandomState(seed)
        self.state = 0

    @abstractmethod
    def start(self):
        self.state = sample(self.d0, self.rng)
        return self.state

    def step(self, a: int):
        p_sp = self.K[self.state, a]
        sp = sample(p_sp, self.rng)

        r = self.reward(self.state, a, sp)
        t = self.terminal(self.state, a, sp)

        self.state = sp

        return (r, self.state, t)
