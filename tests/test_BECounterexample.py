import unittest
import numpy as np
from PyRlEnvs.domains.BECounterexample import BECounterexample, A, B, Bp, behaviorPolicy

np.random.seed(0)

class TestBECounterexample(unittest.TestCase):
    def test_actions(self):
        a = BECounterexample.actions(A)
        self.assertListEqual(list(a), [0, 1])

        a = BECounterexample.actions(B)
        self.assertListEqual(list(a), [0, 1])

        a = BECounterexample.actions(Bp)
        self.assertListEqual(list(a), [0, 1])

    def test_nextStates(self):
        sp = BECounterexample.nextStates(A, 0)
        self.assertListEqual(list(sp), [B])

        sp = BECounterexample.nextStates(A, 1)
        self.assertListEqual(list(sp), [Bp])

        sp = BECounterexample.nextStates(B, 0)
        self.assertListEqual(list(sp), [A])

        sp = BECounterexample.nextStates(B, 1)
        self.assertListEqual(list(sp), [A])

        sp = BECounterexample.nextStates(Bp, 0)
        self.assertListEqual(list(sp), [B])

        sp = BECounterexample.nextStates(Bp, 1)
        self.assertListEqual(list(sp), [Bp])

    def test_reward(self):
        r = BECounterexample.reward(A, 0, B)
        self.assertEqual(r, 0)

        r = BECounterexample.reward(A, 1, Bp)
        self.assertEqual(r, 0)

        r = BECounterexample.reward(B, 0, A)
        self.assertEqual(r, 1)

        r = BECounterexample.reward(B, 1, A)
        self.assertEqual(r, 1)

        r = BECounterexample.reward(Bp, 0, B)
        self.assertEqual(r, -1)

        r = BECounterexample.reward(Bp, 1, Bp)
        self.assertEqual(r, -1)

    def test_terminal(self):
        t = BECounterexample.terminal(A, 0, B)
        self.assertFalse(t)

        t = BECounterexample.terminal(A, 1, Bp)
        self.assertFalse(t)

        t = BECounterexample.terminal(B, 0, A)
        self.assertFalse(t)

        t = BECounterexample.terminal(B, 1, A)
        self.assertFalse(t)

        t = BECounterexample.terminal(Bp, 0, B)
        self.assertFalse(t)

        t = BECounterexample.terminal(Bp, 1, Bp)
        self.assertFalse(t)

    def test_stateful(self):
        env = BECounterexample(0)

        s = env.start()

        r, sp, t, _ = env.step(0)
        self.assertEqual(s, A)
        self.assertEqual(r, 0)
        self.assertEqual(sp, B)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, 1)
        self.assertEqual(sp, A)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, 0)
        self.assertEqual(sp, Bp)
        self.assertFalse(t)

        r, sp, t, _ = env.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(sp, B)
        self.assertFalse(t)

    def test_transitionMatrix(self):
        P = BECounterexample.constructTransitionMatrix(behaviorPolicy)

        expected = np.array([
            [0, .5, .5],
            [1, 0, 0],
            [0, .5, .5],
        ])

        self.assertTrue(np.allclose(P, expected))

        P = BECounterexample.constructTransitionMatrix(behaviorPolicy, gamma=0.9)
        expected = expected * 0.9

        self.assertTrue(np.allclose(P, expected))

    def test_rewardVector(self):
        R = BECounterexample.constructRewardVector(behaviorPolicy)
        expected = np.array([0, 1, -1])

        self.assertTrue(np.allclose(R, expected))
