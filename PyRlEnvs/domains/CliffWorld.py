from PyRlEnvs.Category import addToCategory
import numpy as np
from PyRlEnvs.domains.GridWorld.Elements import Element, StartState, GoalState
from PyRlEnvs.domains.GridWorld.utils import Coords, predecessors
from PyRlEnvs.domains.GridWorld import GridWorldBuilder

class Cliff(Element):
    def __init__(self, loc: Coords):
        super().__init__('Cliff')

        self.loc = loc

    def trigger(self, s: int, a: int, sp: int) -> bool:
        target = self.getState(self.loc)
        return sp == target

    def apply(self, d0: np.ndarray, K: np.ndarray, T: np.ndarray, R: np.ndarray):
        target = self.getState(self.loc)
        starts = np.argwhere(d0 > 0)

        # instead of transitioning into the cliff
        # we need to transition back to the start
        K[:, :, target] = 0
        R[:, :, target] = 0
        T[:, :, target] = 0
        d0[target] = 0

        for (s, a) in predecessors(target, self.shape):
            # reminder that we *don't* ever actually transition into the cliff
            # we end up back in the start state instead
            R[s, a, starts] = -100

            # probably there is only one start state
            # but in case we ever define multiple, then uniform random
            # chance to transition to any of them
            for start in starts:
                K[s, a, start] = 1.0 / len(starts)


def build(shape: Coords = (12, 4)):
    cliffWorldBuilder = GridWorldBuilder(shape)

    # start in bottom left
    cliffWorldBuilder.addElement(StartState((0, 0)))

    # end in bottom right without additional reward
    cliffWorldBuilder.addElement(GoalState((shape[0] - 1, 0), -1))

    # with a cliff in every state in-between
    cliffWorldBuilder.addElements([ Cliff((x, 0)) for x in range(1, shape[0] - 1) ])

    cliffWorldBuilder.costToGoal = True

    return cliffWorldBuilder.build()

CliffWorld = build()

addToCategory('gridworld', CliffWorld)
addToCategory('finite-dynamics', CliffWorld)
addToCategory('sutton-barto', CliffWorld)
