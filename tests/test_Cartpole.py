from typing import Any
import unittest
import numpy as np
from PyRlEnvs.domains.Cartpole import Cartpole
import gym

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

    def test_stateful(self):
        env = Cartpole(seed=0)
        gym_env: Any = gym.make('CartPole-v1')

        gym_env._max_episode_steps = np.inf

        t = False
        s = None
        seed = 0
        for step in range(5000):
            if step % 1000 == 0 or t:
                s_gym, _ = gym_env.reset(seed=seed)
                seed += 1
                s = env.start()
                env._state = s_gym

            a = np.random.choice(env.actions(s))

            r, sp, t = env.step(a)
            sp_gym, r_gym, t_gym, _, _ = gym_env.step(a)

            self.assertTrue(np.allclose(sp, sp_gym))
            self.assertEqual(r, r_gym)
            self.assertEqual(t, t_gym)

            env._state = gym_env.state
