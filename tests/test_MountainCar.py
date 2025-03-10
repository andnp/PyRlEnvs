import unittest
import numpy as np
from rlglue.rl_glue import RlGlue
from PyRlEnvs.domains.MountainCar import GymMountainCar, MountainCar
from tests._utils.toy_agent import ToyAgent

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

    def test_noCheese(self):
        # ensure that the always go right policy doesn't trivially solve this domain
        env = MountainCar()

        for _ in range(50):
            env.start()
            for _ in range(50):
                s, _, t, _, _ = env.step(2)
                self.assertFalse(t)

    def test_bangBang(self):
        # ensure that the bang-bang policy can still solve the problem
        env = MountainCar()

        for _ in range(50):
            s = env.start()
            t = False
            for _ in range(500):
                a = 2 if s[1] >= 0 else 0

                s, _, t, _, _ = env.step(a)
                if t:
                    break

            self.assertTrue(t)

    def test_bangBangRandom(self):
        # ensure that the bang-bang policy can still solve the problem
        # this guarantees that the first 1000 seeds are solvable
        for r in range(1000):
            env = MountainCar(randomize=True, seed=r)
            s = env.start()
            t = False
            for _ in range(500):
                a = 2 if s[1] >= 0 else 0

                s, _, t, _, _ = env.step(a)
                if t:
                    break

            self.assertTrue(t)

    def test_bangBangGym(self):
        # ensure that the bang-bang policy can still solve the problem
        env = GymMountainCar()

        for _ in range(50):
            s = env.start()
            t = False
            for _ in range(500):
                a = 2 if s[1] >= 0 else 0

                s, _, t, _, _ = env.step(a)
                if t:
                    break

            self.assertTrue(t)

    def test_rlglue(self):
        env = GymMountainCar(0)
        agent = ToyAgent(3)

        glue = RlGlue(agent, env)

        glue.start()
        for _ in range(1000):
            interaction = glue.step()
            if interaction.term:
                glue.start()

        # dummy check to ensure we get this far
        self.assertTrue(True)
