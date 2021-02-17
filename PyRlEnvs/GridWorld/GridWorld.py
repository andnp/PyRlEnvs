import numpy as np
from typing import Callable, Sequence, List
from PyRlEnvs.FiniteDynamics import FiniteDynamics
from .Elements import Element
from .utils import Coords, findFirstTrigger, getState, getCoords


class GridWorldBuilder:
    def __init__(self, shape: Coords):
        self.shape = shape
        self.costToGoal = True

        self.elements: List[Element] = []

    def addElement(self, element: Element):
        element.init(self.shape)
        self.elements.append(element)

    def addElements(self, elements: Sequence[Element]):
        for element in elements:
            self.addElement(element)

    def apply(self, s: int, a: int, sp: int, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        for element in self.elements:
            if element.trigger(s, a, sp):
                element.apply(s, a, sp, d0, K, T, R)

    def build(self):
        return buildGridWorld(self)

def buildGridWorld(builder: GridWorldBuilder):
    width, height = builder.shape
    states = width * height
    actions = 4

    # partially apply the getState function
    # to simplify code a bit
    _getState = lambda coords: getState(coords, builder.shape)
    _getCoords = lambda state: getCoords(state, builder.shape)

    # build the dynamics tensors
    _K = np.zeros((states, actions, states))
    _R = np.zeros((states, actions, states))
    _T = np.zeros((states, actions, states))
    _d0 = np.zeros(states)
    for x in range(width):
        for y in range(height):
            s = _getState((x, y))
            for a in range(actions):
                # UP
                if a == 0:
                    sp = _getState((x, y+1))
                # RIGHT
                elif a == 1:
                    sp = _getState((x+1, y))
                # DOWN
                elif a == 2:
                    sp = _getState((x, y-1))
                # LEFT
                else:
                    sp = _getState((x-1, y))

                _K[s, a, sp] = 1

                if builder.costToGoal:
                    _R[s, a, sp] = -1

                # modify the tensors with individual elements
                builder.apply(s, a, sp, _d0, _K, _T, _R)

    # TODO: consider if this should be a warning
    if _d0.sum() == 0:
        _d0[0] = 1.0

    # ensure this ends up as a probability distribution
    _d0 = _d0 / _d0.sum()

    class GridWorld(FiniteDynamics):
        num_states = states
        num_actions = actions

        K=_K
        Rs=_R
        T=_T
        d0 = _d0

        getState: Callable[[Coords], int] = _getState
        getCoords: Callable[[int], Coords] = _getCoords

        @staticmethod
        def show():
            row_str = '-' * (width * 4 + 1)
            print(row_str)
            for y in range(height-1, -1, -1):
                print('|', end='')
                for x in range(width):
                    s = _getState((x, y))
                    element = findFirstTrigger(builder.elements, s, 0, s)
                    if element and element.name:
                        print(f' {element.name[0]} ', end='')
                    else:
                        print('   ', end='')

                    print('|', end='')

                print()
                print(row_str)

    return GridWorld
