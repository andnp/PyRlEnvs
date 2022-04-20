import unittest
import numpy as np
from itertools import product
from PyRlEnvs.domains.GridWorld.utils import getState, predecessor
from PyRlEnvs.domains.GridWorld import GoalState, GridWorldBuilder, StartState, WallState, buildGridWorld

np.random.seed(0)

def testWorld():
    builder = GridWorldBuilder((3, 3))

    builder.addElements([
        StartState((0, 0)),
        GoalState((2, 2), 1)
    ])

    return buildGridWorld(builder)


class TestGridWorld(unittest.TestCase):
    def test_actions(self):
        GridWorld = testWorld()

        actions = GridWorld.actions(0)
        self.assertEqual(list(actions), [0, 1, 2, 3])

    def test_nextStates(self):
        GridWorld = testWorld()

        s_idx = GridWorld.getState((0, 0))
        sp_idx = GridWorld.getState((0, 1))
        sp = GridWorld.nextStates(s_idx, 0)

        self.assertEqual(list(sp), [sp_idx])

        s_idx = GridWorld.getState((0, 0))
        sp_idx = GridWorld.getState((0, 0))
        sp = GridWorld.nextStates(s_idx, 2)

        self.assertEqual(list(sp), [sp_idx])

    def test_reward(self):
        GridWorld = testWorld()

        s = GridWorld.getState((0, 0))
        sp = GridWorld.getState((1, 0))
        r = GridWorld.reward(s, 1, sp)

        self.assertEqual(r, -1)

        s = GridWorld.getState((2, 1))
        sp = GridWorld.getState((2, 2))
        r = GridWorld.reward(s, 0, sp)

        self.assertEqual(r, 1)

    def test_terminal(self):
        GridWorld = testWorld()

        s = GridWorld.getState((0, 0))
        sp = GridWorld.getState((1, 0))
        t = GridWorld.terminal(s, 1, sp)

        self.assertEqual(t, False)

        s = GridWorld.getState((2, 1))
        sp = GridWorld.getState((2, 2))
        t = GridWorld.terminal(s, 0, sp)

        self.assertEqual(t, True)

    def test_stateful(self):
        GridWorld = testWorld()

        env = GridWorld()

        s = env.start()
        self.assertEqual(GridWorld.getCoords(s), (0, 0))

        r, sp, t = env.step(2)
        self.assertEqual(r, -1)
        self.assertEqual(GridWorld.getCoords(sp), (0, 0))
        self.assertEqual(t, False)

        r, sp, t = env.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(GridWorld.getCoords(sp), (0, 1))
        self.assertEqual(t, False)

        r, sp, t = env.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(GridWorld.getCoords(sp), (0, 2))
        self.assertEqual(t, False)

        r, sp, t = env.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(GridWorld.getCoords(sp), (0, 2))
        self.assertEqual(t, False)

        r, sp, t = env.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(GridWorld.getCoords(sp), (1, 2))
        self.assertEqual(t, False)

        r, sp, t = env.step(1)
        self.assertEqual(r, 1)
        self.assertEqual(GridWorld.getCoords(sp), (2, 2))
        self.assertEqual(t, True)

    def test_transitionMatrix(self):
        GridWorld = testWorld()

        def behavior(s: int):
            return np.array([0.25, 0.25, 0.25, 0.25])

        P = GridWorld.constructTransitionMatrix(behavior)

        E = np.zeros((9, 9))
        for s in product(range(3), range(3)):
            s_idx = GridWorld.getState(s)

            # UP
            sp_idx = GridWorld.getState((s[0], s[1] + 1))
            E[s_idx, sp_idx] += 0.25

            # RIGHT
            sp_idx = GridWorld.getState((s[0] + 1, s[1]))
            E[s_idx, sp_idx] += 0.25

            # DOWN
            sp_idx = GridWorld.getState((s[0], s[1] - 1))
            E[s_idx, sp_idx] += 0.25

            # LEFT
            sp_idx = GridWorld.getState((s[0] - 1, s[1]))
            E[s_idx, sp_idx] += 0.25

        self.assertTrue(np.allclose(P, E))

        P = GridWorld.constructTransitionMatrix(behavior, gamma=0.99)
        E *= 0.99
        g_idx = GridWorld.getState((2, 2))
        E[:, g_idx] = 0

        self.assertTrue(np.allclose(P, E))

    def test_walls(self):
        builder = GridWorldBuilder((3, 3))

        builder.addElements([
            StartState((0, 0)),
            GoalState((2, 2), 1)
        ])

        builder.addElement(WallState((1, 0)))

        GridWorld = buildGridWorld(builder)

        env = GridWorld(0)

        s = env.start()
        self.assertEqual(s, GridWorld.getState((0, 0)))

        r, sp, t = env.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(t, False)
        self.assertEqual(sp, GridWorld.getState((0, 0)))

class TestUtils(unittest.TestCase):
    def test_predecessor(self):
        shape = (5, 4)
        sp = getState((3, 3), shape)
        s = predecessor(sp, 0, shape)

        e = sp, getState((3, 2), shape)
        self.assertEqual(s, list(e))

        sp = getState((4, 3), shape)
        s = predecessor(sp, 3, shape)

        self.assertEqual(s, [])
