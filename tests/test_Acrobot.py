from typing import Any
import unittest
import numpy as np
from PyRlEnvs.domains.Acrobot import Acrobot
import gym

np.random.seed(0)

class TestAcrobot(unittest.TestCase):
    def test_actions(self):
        actions = Acrobot.actions(np.zeros(0))
        self.assertListEqual(actions, [0, 1, 2])

    def test_rewards(self):
        r = Acrobot.reward(np.zeros(0), 0, np.zeros(0))
        self.assertEqual(r, -1)

    def test_stateful(self):
        env = Acrobot(0)
        gym_env: Any = gym.make('Acrobot-v1')

        gym_env.seed(0)
        gym_env._max_episode_steps = np.inf

        t = False
        for step in range(5000):
            if step % 1000 == 0 or t:
                gym_env.reset()
                env.start()
                s_ = gym_env.state
                env.setState(s_)

            a = np.random.choice(Acrobot.actions(np.zeros(0)))

            r, sp, t = env.step(a)
            sp_gym, r_gym, t_gym, _ = gym_env.step(a)

            self.assertTrue(np.allclose(sp, sp_gym))
            self.assertEqual(t, t_gym)
            self.assertEqual(r, r_gym)
