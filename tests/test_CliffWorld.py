import unittest
import numpy as np
import PyRlEnvs.domains.CliffWorld as CliffWorld

np.random.seed(0)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class TestCliffWorld(unittest.TestCase):
    def test_stateful(self):
        # make the cliff a bit shorter, just to make the unit test shorter
        Env = CliffWorld.build((5, 4))
        env = Env(seed=0)

        s = env.start()
        self.assertEqual(s, Env.getState((0, 0)))

        r, sp, t, _ = env.step(LEFT)
        self.assertEqual(r, -1)
        self.assertEqual(sp, Env.getState((0, 0)))
        self.assertFalse(t)

        r, sp, t, _ = env.step(RIGHT)
        self.assertEqual(r, -100)
        self.assertEqual(sp, Env.getState((0, 0)))
        self.assertFalse(t)

        r, sp, t, _ = env.step(UP)
        self.assertEqual(r, -1)
        self.assertEqual(sp, Env.getState((0, 1)))
        self.assertFalse(t)

        for x in range(3):
            r, sp, t, _ = env.step(RIGHT)
            self.assertEqual(r, -1)
            self.assertEqual(sp, Env.getState((x + 1, 1)))
            self.assertFalse(t)

        r, sp, t, _ = env.step(DOWN)
        self.assertEqual(r, -100)
        self.assertEqual(sp, Env.getState((0, 0)))
        self.assertFalse(t)

        r, sp, t, _ = env.step(UP)
        self.assertEqual(r, -1)
        self.assertEqual(sp, Env.getState((0, 1)))
        self.assertFalse(t)

        for x in range(4):
            r, sp, t, _ = env.step(RIGHT)
            self.assertEqual(r, -1)
            self.assertEqual(sp, Env.getState((x + 1, 1)))
            self.assertFalse(t)

        r, sp, t, _ = env.step(DOWN)
        self.assertEqual(r, -1)
        self.assertEqual(sp, Env.getState((4, 0)))
        self.assertTrue(t)
