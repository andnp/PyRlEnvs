import unittest
import numpy as np
from PyRlEnvs.domains.BairdCounterexample import BairdCounterexample, targetPolicy, behaviorPolicy

np.random.seed(0)

class TestBairdCounterexample(unittest.TestCase):
    def test_actions(self):
        actions = BairdCounterexample.actions(0)
        self.assertEqual(list(actions), [0, 1])

    def test_nextStates(self):
        sp = BairdCounterexample.nextStates(0, 0)
        self.assertEqual(list(sp), [0, 1, 2, 3, 4, 5])

        sp = BairdCounterexample.nextStates(2, 1)
        self.assertEqual(list(sp), [6])

    def test_reward(self):
        r = BairdCounterexample.reward(0, 0, 1)
        self.assertEqual(r, 0)

    def test_terminal(self):
        t = BairdCounterexample.terminal(0, 0, 1)
        self.assertFalse(t)

    def test_stateful(self):
        env = BairdCounterexample(0)

        s = env.start()
        self.assertEqual(s, 6)

        r, sp, t = env.step(0)
        self.assertEqual(sp, 1)
        self.assertEqual(r, 0)
        self.assertFalse(t)

        r, sp, t = env.step(1)
        self.assertEqual(sp, 6)
        self.assertEqual(r, 0)
        self.assertFalse(t)

    def test_transitionMatrix(self):
        P = BairdCounterexample.constructTransitionMatrix(targetPolicy)
        E = np.zeros((7, 7))
        E[:, 6] = 1

        self.assertTrue(np.allclose(P, E))

        P = BairdCounterexample.constructTransitionMatrix(behaviorPolicy)
        E = np.zeros((7, 7))
        E[:, :] = 1 / 7

        self.assertTrue(np.allclose(P, E))

    def test_rewardVector(self):
        R = BairdCounterexample.constructRewardVector(targetPolicy)
        E = np.zeros(7)

        self.assertTrue(np.allclose(R, E))
