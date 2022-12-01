import unittest
import numpy as np
from PyRlEnvs.domains.Acrobot import Acrobot

np.random.seed(0)

class TestAcrobot(unittest.TestCase):
    def test_actions(self):
        env = Acrobot()
        actions = env.actions(np.zeros(0))
        self.assertListEqual(actions, [0, 1, 2])

    def test_rewards(self):
        env = Acrobot()
        r = env.reward(np.zeros(0), 0, np.zeros(0))
        self.assertEqual(r, -1)

    # TODO: this fails with modern gym code, they've made several changes
    # to the rng and floating-point precision
    # def test_stateful(self):
    #     env = Acrobot(seed=0)
    #     rng = np.random.default_rng(0)

    #     t = False
    #     s = None
    #     with open('tests/data/test_acrobot.pkl', 'rb') as f:
    #         expected = pickle.load(f)

    #     for step in range(5000):
    #         if step % 521 == 0 or t:
    #             s = env.start()

    #         a = choice(env.actions(np.zeros(0)), rng)
    #         r, sp, t = env.step(a)

    #         s = sp

    #         es, ea, er, esp, et = expected[step]
    #         self.assertTrue(np.all(s == es))
    #         self.assertEqual(a, ea)
    #         self.assertEqual(r, er)
    #         self.assertTrue(np.all(sp == esp))
    #         self.assertEqual(t, et)
