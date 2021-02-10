import unittest
import numpy as np
from PyRlEnvs.BECounterexample import BECounterexample, A, B, Bp

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
        self.assertListEqual(sp, [B])

        sp = BECounterexample.nextStates(A, 1)
        self.assertListEqual(sp, [Bp])

        sp = BECounterexample.nextStates(B, 0)
        self.assertListEqual(sp, [A])

        sp = BECounterexample.nextStates(B, 1)
        self.assertListEqual(sp, [A])

        sp = BECounterexample.nextStates(Bp, 0)
        self.assertListEqual(sp, [B])

        sp = BECounterexample.nextStates(Bp, 1)
        self.assertListEqual(sp, [Bp])

    def test_reward(self):
        pass

    def test_terminal(self):
        pass

    def test_stateful(self):
        pass

    def test_transitionMatrix(self):
        pass

    def test_rewardVector(self):
        pass
