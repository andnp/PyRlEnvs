import numpy as np
from PyRlEnvs.Category import addToCategory
from PyRlEnvs.domains.GridWorld.Elements import Element, WallState
from PyRlEnvs.domains.GridWorld import GridWorldBuilder

class UniformStart(Element):
    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        for s in range(K.shape[0]):
            d0[s] = 1

def build():
    fourRoomsBuilder = GridWorldBuilder((11, 11))
    fourRoomsBuilder.addElement(UniformStart())
    fourRoomsBuilder.addElements([
        # Split top/bottom left
        WallState((0, 5)),
        WallState((2, 5)),
        WallState((3, 5)),
        WallState((4, 5)),
        WallState((5, 5)),
        # Split top left/right
        WallState((5, 10)),
        WallState((5, 9)),
        WallState((5, 7)),
        WallState((5, 6)),
        # Split bottom left/right
        WallState((5, 4)),
        WallState((5, 3)),
        WallState((5, 2)),
        WallState((5, 0)),
        # Split top/bottom right
        WallState((6, 4)),
        WallState((7, 4)),
        WallState((9, 4)),
        WallState((10, 4)),
    ])

    return fourRoomsBuilder.build()

FourRooms = build()

addToCategory('gridworld', FourRooms)
addToCategory('finite-dynamics', FourRooms)
