import numpy as np
from PyRlEnvs.utils.math import try2FastJit
import PyRlEnvs.utils.random as Random
from typing import Any, List, Set
from PyRlEnvs.FiniteDynamics import FiniteDynamics
from PyRlEnvs.domains.GridWorld.utils import Coords, getCoords, getState

# -------------------
# Maze generating alg
# -------------------

def sample(_shape: Coords, costToGoal: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)

    # collect some metadata
    width, height = _shape
    states = width * height

    # build the environment transition kernels
    _K = np.zeros((states, 4, states))
    _R = np.zeros((states, 4, states))
    _T = np.zeros((states, 4, states))
    _d0 = np.zeros(states)

    # build the set of states that the wall-builder
    # has not yet visited, initially this is every state
    unvisited = set(range(states))

    # pick a state and mark it as visited
    # we will guarantee that all paths connect to this state
    # at some point
    start = Random.choice(unvisited, rng)
    unvisited.remove(start)

    # the terminal state is in the top right of the maze
    terminal_state = getState((width - 1, height - 1), _shape)
    _T[terminal_state, :, terminal_state] = 1

    # build a simple -1 per step reward function
    r = -1 if costToGoal else 0
    rt = -1 if costToGoal else 1

    # sample paths until we've visited every state
    while len(unvisited) > 0:
        path = _samplePath(unvisited, _shape, rng)

        # walk the path and "activate" every transition from
        # prev -> cell. need to carefully handle termination states
        # also make sure the agent can walk backwards through the space
        # by connecting cell -> prev
        prev = None
        for cell in path:
            if prev is not None:
                unvisited.remove(prev)

                # mark all available actions from prev -> cell
                # might be multiple actions due to bumping into walls
                for a in actions(prev, cell, _shape):
                    _K[prev, a, cell] = 1
                    _R[prev, a, cell] = r

                    if cell == terminal_state:
                        _T[prev, a, cell] = 1
                        _R[prev, a, cell] = rt

                # mark all available actions from cell -> prev
                for a in actions(cell, prev, _shape):
                    _K[cell, a, prev] = 1
                    _R[cell, a, prev] = r

                    if prev == terminal_state:
                        _T[cell, a, prev] = 1
                        _R[cell, a, prev] = rt

            prev = cell

    # now we need to make sure all self-connections exist
    # that is, if I run into a wall then I stay in the same state
    for state in range(states):
        for a in range(4):
            # if this action doesn't lead anywhere, then it needs to be a self-transition
            if _K[state, a].sum() == 0:
                _K[state, a, state] = 1
                _R[state, a, state] = r

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

@try2FastJit
def neighbors(state: int, shape: Coords):
    x, y = getCoords(state, shape)

    return set([
        getState((x + 1, y), shape),
        getState((x - 1, y), shape),
        getState((x, y + 1), shape),
        getState((x, y - 1), shape),
    ])

@try2FastJit
def actions(state: int, next_state: int, shape: Coords):
    x, y = getCoords(state, shape)

    # trick numba into know the type of this empty list
    ret: List[int] = [i for i in range(0)]

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

        # delete loops, otherwise we cannot guarantee solvability
        if cell in path:
            path = path[0:path.index(cell) + 1]
        else:
            path.append(cell)

    return path

# -----------------------
# Provide a default class
# -----------------------
Maze10x10 = sample((10, 10))
Maze5x5 = sample((5, 5))
