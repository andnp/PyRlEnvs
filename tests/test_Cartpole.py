import unittest
import numpy as np
from PyRlEnvs.domains.Cartpole import Cartpole

np.random.seed(0)

class TestCartpole(unittest.TestCase):
    def test_actions(self):
        env = Cartpole(seed=0)
        actions = env.actions(np.zeros(0))
        self.assertListEqual(actions, [0, 1])

    def test_rewards(self):
        env = Cartpole(seed=0)
        r = env.reward(np.zeros(0), 0, np.zeros(0))
        self.assertEqual(r, 1)
