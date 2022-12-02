import unittest
import numpy as np
from PyRlEnvs.domains.PuddleWorld import PuddleWorld, UP, RIGHT

np.random.seed(0)

class TestPuddleWorld(unittest.TestCase):
    def test_actions(self):
        self.assertEqual(PuddleWorld.actions(0), [0, 1, 2, 3])

    def test_stateful(self):
        env = PuddleWorld(seed=0)

        s = env.start()
        self.assertTrue(np.allclose(s, [0.06257302, 0.03678951]))

        r, s, t, _ = env.step(UP)
        self.assertEqual(r, -1)
        self.assertTrue(np.allclose(s, [0.06257302, 0.08804682]))

        for _ in range(15):
            r, s, t, _ = env.step(UP)

        for _ in range(4):
            r, s, t, _ = env.step(RIGHT)

        self.assertTrue(np.allclose(r, -23.31279899231159))
        self.assertTrue(np.allclose(s, [0.26850887, 0.794218]))
