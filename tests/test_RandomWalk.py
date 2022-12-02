import unittest
import numpy as np
from PyRlEnvs.domains.RandomWalk import RandomWalk, dependentFeatures, invertedFeatures

np.random.seed(0)

class TestRandomWalk(unittest.TestCase):
    def test_actions(self):
        actions = RandomWalk.actions(0)
        self.assertListEqual(list(actions), [0, 1])

    def test_nextStates(self):
        sps = RandomWalk.nextStates(0, 0)
        self.assertListEqual(list(sps), [2])

        sps = RandomWalk.nextStates(2, 1)
        self.assertListEqual(list(sps), [3])

        sps = RandomWalk.nextStates(4, 1)
        self.assertListEqual(list(sps), [2])

    def test_reward(self):
        r = RandomWalk.reward(3, 0, 2)
        self.assertEqual(r, 0)

        r = RandomWalk.reward(4, 1, 2)
        self.assertEqual(r, 1)

        r = RandomWalk.reward(4, 0, 3)
        self.assertEqual(r, 0)

        r = RandomWalk.reward(0, 0, 2)
        self.assertEqual(r, -1)

        r = RandomWalk.reward(0, 1, 1)
        self.assertEqual(r, 0)

    def test_terminal(self):
        t = RandomWalk.terminal(0, 0, 2)
        self.assertEqual(t, True)

        t = RandomWalk.terminal(0, 1, 1)
        self.assertEqual(t, False)

        t = RandomWalk.terminal(1, 1, 2)
        self.assertEqual(t, False)

        t = RandomWalk.terminal(4, 0, 3)
        self.assertEqual(t, False)

        t = RandomWalk.terminal(4, 1, 2)
        self.assertEqual(t, True)

    def test_stateful(self):
        env = RandomWalk(0)

        s = env.start()
        self.assertEqual(s, 2)

        r, s, t, _ = env.step(1)
        self.assertEqual(s, 3)
        self.assertEqual(r, 0)
        self.assertEqual(t, False)

        r, s, t, _ = env.step(1)
        self.assertEqual(s, 4)
        self.assertEqual(r, 0)
        self.assertEqual(t, False)

        r, s, t, _ = env.step(1)
        self.assertEqual(s, 2)
        self.assertEqual(r, 1)
        self.assertEqual(t, True)

    def test_transitionMatrix(self):
        policy = lambda s: np.array([0.25, 0.75])
        P = RandomWalk.constructTransitionMatrix(policy)

        expected = np.array([
            [0, 0.75, 0.25, 0, 0],
            [0.25, 0, 0.75, 0, 0],
            [0, 0.25, 0, 0.75, 0],
            [0, 0, 0.25, 0, 0.75],
            [0, 0, 0.75, 0.25, 0],
        ])

        self.assertTrue(np.allclose(P, expected))

        P = RandomWalk.constructTransitionMatrix(policy, gamma=0.9)

        expected = np.array([
            [0, 0.75, 0, 0, 0],
            [0.25, 0, 0.75, 0, 0],
            [0, 0.25, 0, 0.75, 0],
            [0, 0, 0.25, 0, 0.75],
            [0, 0, 0, 0.25, 0],
        ]) * 0.9

        self.assertTrue(np.allclose(P, expected))

    def test_rewardVector(self):
        policy = lambda s: np.array([0.25, 0.75])

        R = RandomWalk.constructRewardVector(policy)
        expected = np.array([-0.25, 0, 0, 0, 0.75])

        self.assertTrue(np.allclose(R, expected))

    def test_stateDist(self):
        policy = lambda s: np.array([.5, .5])

        d = RandomWalk.computeStateDistribution(policy)
        expected = np.array([1, 2, 3, 2, 1]) / 9

        self.assertTrue(np.allclose(d, expected))

    def test_features(self):
        got = dependentFeatures(5)
        expected = np.array([
            [1,     0,    0], # noqa: E241, E226
            [1/2, 1/2,    0], # noqa: E241, E226
            [1/3, 1/3,  1/3], # noqa: E241, E226
            [0,   1/2,  1/2], # noqa: E241, E226
            [0,     0,    1], # noqa: E241, E226
        ])

        self.assertTrue(np.allclose(got, expected))

        got = invertedFeatures(5)
        expected = np.array([
            [0,    1/4,  1/4,  1/4,  1/4], # noqa: E241, E226
            [1/4,    0,  1/4,  1/4,  1/4], # noqa: E241, E226
            [1/4,  1/4,    0,  1/4,  1/4], # noqa: E241, E226
            [1/4,  1/4,  1/4,    0,  1/4], # noqa: E241, E226
            [1/4,  1/4,  1/4,  1/4,    0], # noqa: E241, E226
        ])

        self.assertTrue(np.allclose(got, expected))
