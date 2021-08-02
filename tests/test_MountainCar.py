from typing import Any
import unittest
import numpy as np
from PyRlEnvs.domains.MountainCar import GymMountainCar, MountainCar
import gym

np.random.seed(0)

class TestMountain(unittest.TestCase):
    def test_actions(self):
        env = MountainCar()
        actions = env.actions(np.zeros(0))
        self.assertListEqual(actions, [0, 1, 2])

    def test_rewards(self):
        env = MountainCar()
        r = env.reward(np.zeros(0), 0, np.zeros(0))
        self.assertEqual(r, -1)

    def test_nextState(self):
        s = np.array([0, 0])
        a = 0

        env = GymMountainCar()
        sp = env.nextState(s, a)
        self.assertTrue(np.allclose(sp, np.array([-0.0035, -0.0035])))

        gym_env: Any = gym.make('MountainCar-v0')

        for _ in range(1000):
            s = gym_env.reset()
            a = np.random.choice(env.actions(s))

            sp = env.nextState(s, a)

            sp_gym, r_gym, _, _ = gym_env.step(a)

            self.assertTrue(np.allclose(sp, sp_gym))
            self.assertEqual(env.reward(s, a, sp), r_gym)

    def test_stateful(self):
        env = GymMountainCar(0)
        gym_env: Any = gym.make('MountainCar-v0')

        gym_env.seed(0)
        gym_env._max_episode_steps = np.inf

        t = False
        s = None
        for step in range(5000):
            if step % 1000 == 0 or t:
                s_gym = gym_env.reset()
                s = env.start()
                env._state = s_gym

            a = np.random.choice(env.actions(s))

            r, sp, t = env.step(a)
            sp_gym, r_gym, t_gym, _ = gym_env.step(a)

            self.assertTrue(np.allclose(sp, sp_gym))
            self.assertEqual(r, r_gym)
            self.assertEqual(t, t_gym)
