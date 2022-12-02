from PyRlEnvs.domains.BoyanChain import BoyanChain, behaviorPolicy
import unittest
import numpy as np

np.random.seed(0)

class TestBoyanChain(unittest.TestCase):
    def test_actions(self):
        actions = BoyanChain.actions(0)

        self.assertEqual(list(actions), [0, 1])

    def test_nextStates(self):
        sp = BoyanChain.nextStates(0, 0)
        self.assertEqual(list(sp), [1])

        sp = BoyanChain.nextStates(0, 1)
        self.assertEqual(list(sp), [2])

        sp = BoyanChain.nextStates(11, 0)
        self.assertEqual(list(sp), [12])

        sp = BoyanChain.nextStates(12, 0)
        self.assertEqual(list(sp), [0])

    def test_reward(self):
        reward = BoyanChain.reward(0, 0, 1)
        self.assertEqual(reward, -3)

        reward = BoyanChain.reward(2, 1, 4)
        self.assertEqual(reward, -3)

        reward = BoyanChain.reward(11, 0, 12)
        self.assertEqual(reward, -2)

        reward = BoyanChain.reward(12, 0, 0)
        self.assertEqual(reward, 0)

    def test_terminal(self):
        t = BoyanChain.terminal(0, 0, 1)
        self.assertFalse(t)

        t = BoyanChain.terminal(12, 0, 0)
        self.assertTrue(t)

    def test_stateful(self):
        env = BoyanChain(0)

        s = env.start()
        self.assertEqual(s, 0)

        r, sp, t, _ = env.step(0)
        self.assertEqual(r, -3)
        self.assertEqual(sp, 1)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, -3)
        self.assertEqual(sp, 3)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, -3)
        self.assertEqual(sp, 5)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, -3)
        self.assertEqual(sp, 7)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, -3)
        self.assertEqual(sp, 9)
        self.assertFalse(t)

        r, sp, t, _ = env.step(1)
        self.assertEqual(r, -3)
        self.assertEqual(sp, 11)
        self.assertFalse(t)

        r, sp, t, _ = env.step(0)
        self.assertEqual(r, -2)
        self.assertEqual(sp, 12)
        self.assertFalse(t)

        r, sp, t, _ = env.step(0)
        self.assertEqual(r, 0)
        self.assertEqual(sp, 0)
        self.assertTrue(t)

    def test_transitionMatrix(self):
        P = BoyanChain.constructTransitionMatrix(behaviorPolicy)

        E = np.zeros((13, 13))
        for i in range(11):
            E[i, i + 1] = 0.5
            E[i, i + 2] = 0.5

        E[11, 12] = 1
        E[12, 0] = 1

        self.assertTrue(np.allclose(P, E))

        P = BoyanChain.constructTransitionMatrix(behaviorPolicy, gamma=1)
        E[12] = 0

        self.assertTrue(np.allclose(P, E))

    def test_rewardVector(self):
        R = BoyanChain.constructRewardVector(behaviorPolicy)
        E = np.zeros(13)
        E[:11] = -3
        E[11] = -2

        self.assertTrue(np.allclose(R, E))

    def test_vpi(self):
        P = BoyanChain.constructTransitionMatrix(behaviorPolicy, gamma=1)
        R = BoyanChain.constructRewardVector(behaviorPolicy)

        I = np.eye(P.shape[0])

        vpi = np.linalg.pinv(I - P).dot(R)
        e = np.array([-24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0.])

        self.assertTrue(np.allclose(vpi, e))
