from PyRlEnvs.FiniteDynamics import FiniteDynamics
from typing import List, TypeVar
import numpy as np
import PyRlEnvs.utils.random as Random
from PyRlEnvs.GridWorld.utils import Coords, getCoords, getState
from PyRlEnvs.GridWorld.GridWorld import buildKernels

T = TypeVar('T')

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

def sample(shape: Coords, seed: int = 0):
    rng = np.random.RandomState(seed)

    width, height = shape
    states = width * height

    _K = np.zeros((states, 4, states))
    _R = np.zeros((states, 4, states))
    _T = np.zeros((states, 4, states))
    _d0 = np.zeros(states)

    unvisited = set(range(states))

    start = Random.choice(unvisited, rng)
    unvisited.remove(start)

    while len(unvisited) > 0:
        cell = Random.choice(unvisited, rng)

        path = [cell]

        while cell in unvisited:
            cell = Random.choice(neighbors(cell, shape), rng)

            if cell in path:
                path = path[0:path.index(cell) + 1]
            else:
                path.append(cell)

        prev = None
        for cell in path:
            if prev is not None:
                unvisited.remove(prev)

                for a in actions(prev, cell, shape):
                    _K[prev, a, cell] = 1
                    _R[prev, a, cell] = -1

                for a in actions(cell, prev, shape):
                    _K[cell, a, prev] = 1
                    _R[prev, a, cell] = -1

            prev = cell

    # now we need to make sure all self-connections exist
    # that is, if I run into a wall then I stay in the same state
    for state in range(states):
        for a in range(4):
            # if this action doesn't lead anywhere, then it needs to be a self-transition
            if _K[state, a].sum() == 0:
                _K[state, a, state] = 1
                _R[state, a, state] = -1

    class WilsonMaze(FiniteDynamics):
        num_states = states
        num_actions = 4

        K = _K
        Rs = _R
        T = _T
        d0 = _d0

        @staticmethod
        def getState(coords: Coords):
            return getState(coords, shape)

    return WilsonMaze

WIDTH = 5
HEIGHT = 5
maze = sample((WIDTH, HEIGHT), 2)

tops: List[List[str]] = []
rlines: List[List[str]] = []
llines: List[List[str]] = []
bottoms: List[List[str]] = []
for y in range(HEIGHT):
    top: List[str] = []
    rline: List[str] = []
    bottom: List[str] = []
    lline: List[str] = []
    for x in range(WIDTH):
        state = maze.getState((x, y))

        if maze.K[state, 1, state] == 0:
            rline.append(' ')
        else:
            rline.append('|')

        if maze.K[state, 0, state] == 0:
            top.append(' ')
        else:
            top.append('-')

        if maze.K[state, 2, state] == 0:
            bottom.append(' ')
        else:
            bottom.append('-')

        if maze.K[state, 3, state] == 0:
            lline.append(' ')
        else:
            lline.append('|')

    rline.append('|')
    top.append('|')
    bottom.append('|')
    lline.append('|')

    tops.append(top)
    rlines.append(rline)
    llines.append(lline)
    bottoms.append(bottom)

for y in range(HEIGHT - 1, -1, -1):
    for x in range(WIDTH):
        print('+' + tops[y][x] * 3, end='')
    print('+')

    for x in range(WIDTH):
        print(f'{llines[y][x]}   ', end='')
    print('|')

print('+---' * WIDTH + '+')
