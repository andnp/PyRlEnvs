import numpy as np
from numba import njit
from typing import Callable, Optional, Sequence, cast
from PyRlEnvs.utils.RandomVariables import DiscreteRandomVariable
from PyRlEnvs.BaseEnvironment import BaseEnvironment
from PyRlEnvs.utils.random import sample

class FiniteDynamics(BaseEnvironment):
    # start state dist
    d0 = np.zeros(0)

    # transition kernel
    # shape: (states, actions, states)
    K = np.zeros(())

    # reward kernel
    # shape: (states, actions, states)
    Rs = np.zeros(())

    # termination kernel
    # shape: (states, actions, states)
    T = np.zeros(())

    num_states = 0
    num_actions = 0

    # ----------------------
    # -- RLGlue Interface --
    # ----------------------

    def __init__(self, seed: int = 0):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.state: int = 0

    def start(self):
        self.state = sample(self.d0, self.rng)
        return self.state

    def step(self, action: int):
        sp = self.nextStates(self.state, action).sample(self.rng)
        r = self.reward(self.state, action, sp)
        t = self.terminal(self.state, action, sp)

        gamma = 0.0 if t else 1.0

        self.state = sp

        return (r, self.state, t, { 'gamma': gamma })

    def setState(self, state: int):
        self.state = state

    def copy(self):
        c = self.__class__(self._seed)
        c.state = self.state

        return c

    # -------------------------
    # -- Stateless Interface --
    # -------------------------

    @classmethod
    def actions(cls, s: int):
        # the available actions are any action where the kernel is non-zero for the given state
        return _actions(cls.K, s)

    @classmethod
    def nextStates(cls, s: int, a: int):
        sp = _nextStates(cls.K, s, a)
        sp = cast(Sequence[int], sp)
        return DiscreteRandomVariable(sp, cls.K[s, a, sp])

    @classmethod
    def reward(cls, s: int, a: int, sp: int) -> float:
        return cls.Rs[s, a, sp]

    @classmethod
    def terminal(cls, s: int, a: int, sp: int):
        return bool(cls.T[s, a, sp])

    @classmethod
    def constructTransitionMatrix(cls, policy: Callable[[int], np.ndarray], gamma: Optional[float] = None):
        if gamma is None:
            gamma = -1

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

# ----------------------------
# -- Jit Compiled Utilities --
# ----------------------------

@njit(cache=True)
def _actions(K: np.ndarray, s: int):
    return np.unique(np.where(K[s] != 0)[0])

@njit(cache=True)
def _nextStates(K: np.ndarray, s: int, a: int):
    return np.where(K[s, a] > 0)[0]

@njit(cache=True)
def _transitionMatrix(K: np.ndarray, T: np.ndarray, pi: np.ndarray, gamma: float):
    states = pi.shape[0]
    P = np.zeros((states, states))

    g: np.ndarray = (1 - T) * gamma
    if gamma < 0:
        g = np.ones_like(T)

    for s in range(states):
        for sp in range(states):
            P[s, sp] = np.sum(K[s, :, sp] * pi[s] * g[s, :, sp])

    return P

@njit(cache=True)
def _averageReward(K: np.ndarray, Rs: np.ndarray, pi: np.ndarray):
    states = pi.shape[0]

    R = np.zeros(states)
    for s in range(states):
        for sp in range(states):
            R[s] += np.sum(K[s, :, sp] * Rs[s, :, sp] * pi[s])

    return R

@njit(cache=True)
def _stateDistribution(P: np.ndarray):
    return np.linalg.matrix_power(P, 1024).sum(axis=0) / P.shape[0]
