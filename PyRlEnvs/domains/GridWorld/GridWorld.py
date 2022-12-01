import numpy as np
from typing import Sequence, List
from PyRlEnvs.FiniteDynamics import FiniteDynamics
from .Elements import Element
from .utils import Coords, findFirstTrigger, getState, getCoords

# --------------
# Static builder
# --------------

# To make sure we can build gridworlds statically
# use a builder class to collect all of the meta-data
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

    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        for element in self.elements:
            element.apply(d0, K, T, R)

    def build(self):
        return buildGridWorld(self)


# ----------------------
# Common gridworld logic
# ----------------------

# To make sure the factory function does not get too complex
# pull shared logic out into its own base class
class _BaseGridWorld(FiniteDynamics):
    shape: Coords
    elements: List[Element]

    @classmethod
    def getState(cls, c: Coords) -> int:
        return getState(c, cls.shape)

    @classmethod
    def getCoords(cls, s: int) -> Coords:
        return getCoords(s, cls.shape)

    def addElement(self, element: Element):
        element.init(self.shape)
        element.apply(self.d0, self.K, self.T, self.Rs)
        self.elements.append(element)

    def addElements(self, elements: List[Element]):
        for element in elements:
            self.addElement(element)

    @classmethod
    def show(cls):
        width, height = cls.shape
        row_str = '+---' * (width) + '+'
        print(row_str)
        for y in range(height - 1, -1, -1):
            print('|', end='')
            for x in range(width):
                s = cls.getState((x, y))
                element = findFirstTrigger(cls.elements, s, 0, s)
                if element and element.name:
                    print(f' {element.name[0]} ', end='')
                else:
                    print('   ', end='')

                print('|', end='')

            print()
            print(row_str)

# generate the basic transition and reward kernels for an empty gridworld
# we can later modify these when we add new elements
def _buildKernels(shape: Coords, costToGoal: bool):
    width, height = shape
    states = width * height

    def _getState(coords: Coords):
        return getState(coords, shape)

    _K = np.zeros((states, 4, states))
    _R = np.zeros((states, 4, states))

    for x in range(width):
        for y in range(height):
            s = _getState((x, y))
            for a in range(4):
                # UP
                if a == 0:
                    sp = _getState((x, y + 1))
                # RIGHT
                elif a == 1:
                    sp = _getState((x + 1, y))
                # DOWN
                elif a == 2:
                    sp = _getState((x, y - 1))
                # LEFT
                else:
                    sp = _getState((x - 1, y))

                _K[s, a, sp] = 1

                if costToGoal:
                    _R[s, a, sp] = -1

    return _K, _R

# --------------------------
# Gridworld Factory Function
# --------------------------
def buildGridWorld(builder: GridWorldBuilder):
    width, height = builder.shape
    states = width * height
    actions = 4

    # partially apply the getState function
    # to simplify code a bit
    def _getState(coords: Coords):
        return getState(coords, builder.shape)

    def _getCoords(state: int):
        return getCoords(state, builder.shape)

    # build the dynamics tensors
    _T = np.zeros((states, actions, states))
    _d0 = np.zeros(states)

    _K, _R = _buildKernels(builder.shape, builder.costToGoal)

    # modify the tensors with individual elements
    builder.apply(_d0, _K, _T, _R)

    # TODO: consider if this should be a warning
    if _d0.sum() == 0:
        _d0[0] = 1.0

    # ensure this ends up as a probability distribution
    _d0 = _d0 / _d0.sum()

    class GridWorld(_BaseGridWorld):
        shape = builder.shape
        elements = builder.elements

        num_states = states
        num_actions = actions

        K = _K
        Rs = _R
        T = _T
        d0 = _d0

    return GridWorld
