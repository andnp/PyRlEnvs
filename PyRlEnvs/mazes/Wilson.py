from PyRlEnvs.FiniteDynamics import FiniteDynamics
from typing import Any, List, Set
import numpy as np
import PyRlEnvs.utils.random as Random
from PyRlEnvs.GridWorld.utils import Coords, getCoords, getState

# -------------------
# Maze generating alg
# -------------------

def sample(_shape: Coords, seed: int = 0):
    rng = np.random.RandomState(seed)

    width, height = _shape
    states = width * height

    _K = np.zeros((states, 4, states))
    _R = np.zeros((states, 4, states))
    _T = np.zeros((states, 4, states))
    _d0 = np.zeros(states)

    unvisited = set(range(states))

    start = Random.choice(unvisited, rng)
    unvisited.remove(start)

    terminal_state = getState((width - 1, height - 1), _shape)
    _T[terminal_state, :, terminal_state] = 1

    while len(unvisited) > 0:
        path = _samplePath(unvisited, _shape, rng)

        prev = None
        for cell in path:
            if prev is not None:
                unvisited.remove(prev)

                for a in actions(prev, cell, _shape):
                    _K[prev, a, cell] = 1
                    _R[prev, a, cell] = -1

                    if cell == terminal_state:
                        _T[prev, a, cell] = 1

                for a in actions(cell, prev, _shape):
                    _K[cell, a, prev] = 1
                    _R[cell, a, prev] = -1

                    if prev == terminal_state:
                        _T[cell, a, prev] = 1

            prev = cell

    # now we need to make sure all self-connections exist
    # that is, if I run into a wall then I stay in the same state
    for state in range(states):
        for a in range(4):
            # if this action doesn't lead anywhere, then it needs to be a self-transition
            if _K[state, a].sum() == 0:
                _K[state, a, state] = 1
                _R[state, a, state] = -1

    # set start state as the bottom left
    start = getState((0, 0), _shape)
    _d0[start] = 1

    class WilsonMaze(_WilsonMaze):
        shape = _shape

        num_states = states
        num_actions = 4

        K = _K
        Rs = _R
        T = _T
        d0 = _d0

    return WilsonMaze

# ------------------------
# Internal utility methods
# ------------------------

class _WilsonMaze(FiniteDynamics):
    shape: Coords

    @classmethod
    def getState(cls, coords: Coords):
        return getState(coords, cls.shape)

    @classmethod
    def getCoords(cls, state: int):
        return getCoords(state, cls.shape)

    @classmethod
    def show(cls):
        width, height = cls.shape
        tops: List[List[str]] = []
        sides: List[List[str]] = []

        for y in range(height):
            top: List[str] = []
            side: List[str] = []
            for x in range(width):
                state = cls.getState((x, y))

                if cls.K[state, 0, state] == 0:
                    top.append(' ')
                else:
                    top.append('-')

                if cls.K[state, 3, state] == 0:
                    side.append(' ')
                else:
                    side.append('|')

            top.append('|')
            side.append('|')

            tops.append(top)
            sides.append(side)

        for y in range(height - 1, -1, -1):
            for x in range(width):
                print('+' + tops[y][x] * 3, end='')
            print('+')

            for x in range(width):
                print(f'{sides[y][x]}   ', end='')
            print('|')

        print('+---' * width + '+')

def neighbors(state: int, shape: Coords):
    x, y = getCoords(state, shape)

    return set([
        getState((x + 1, y), shape),
        getState((x - 1, y), shape),
        getState((x, y + 1), shape),
        getState((x, y - 1), shape),
    ])

def actions(state: int, next_state: int, shape: Coords):
    x, y = getCoords(state, shape)
    ret: List[int] = []

    up = getState((x, y + 1), shape)
    if up == next_state:
        ret.append(0)

    right = getState((x + 1, y), shape)
    if right == next_state:
        ret.append(1)

    down = getState((x, y - 1), shape)
    if down == next_state:
        ret.append(2)

    left = getState((x - 1, y), shape)
    if left == next_state:
        ret.append(3)

    return ret

def _samplePath(unvisited: Set[int], shape: Coords, rng: Any):
    cell = Random.choice(unvisited, rng)

    path = [cell]

    while cell in unvisited:
        cell = Random.choice(neighbors(cell, shape), rng)

        if cell in path:
            path = path[0:path.index(cell) + 1]
        else:
            path.append(cell)

    return path
