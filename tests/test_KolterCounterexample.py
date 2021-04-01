import unittest
import numpy as np
from PyRlEnvs.domains.KolterCounterexample import KolterCounterexample, buildProblem

np.random.seed(0)

class TestKolterCounterexample(unittest.TestCase):
    def test_actions(self):
        env = KolterCounterexample(0)
        actions = env.actions(0)
        self.assertEqual(list(actions), [0, 1])

    def test_transitionMatrix(self):
        behavior, _, _, gamma = buildProblem(0.01, 0.5)

        P = KolterCounterexample.constructTransitionMatrix(behavior, gamma=gamma)
        expected = np.ones((2, 2)) * 0.5 * gamma

        self.assertTrue(np.allclose(P, expected))

    def test_vstar(self):
        _, target, _, gamma = buildProblem(0.01)

        P = KolterCounterexample.constructTransitionMatrix(target, gamma=gamma)
        R = KolterCounterexample.constructRewardVector(target)
        I = np.eye(P.shape[0])

        v_star = np.linalg.pinv(I - P).dot(R)

        self.assertTrue(np.allclose(v_star, [1, 1.05]))
