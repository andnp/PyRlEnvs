import unittest
import numpy as np
from PyRlEnvs.domains.HIVTreatment import HIVTreatment

np.random.seed(0)

# TODO: these aren't real tests.
# At best they are sanity checks to make sure the api is correct
# and that the integrator is stable
class TestHIV(unittest.TestCase):
    def test_actions(self):
        actions = HIVTreatment.actions(np.zeros(6))
        self.assertListEqual(actions, [0, 1, 2, 3])

    def test_rewards(self):
        r = HIVTreatment.reward(np.zeros(6), 0, np.zeros(6))
        self.assertEqual(r, 0)

    def test_stateful(self):
        env = HIVTreatment(seed=0)

        s = env.start()
        expected = np.log10([163573, 5., 11945, 46, 63919, 24])
        self.assertTrue(np.allclose(s, expected))

        r, sp, t, _ = env.step(2)
        self.assertTrue(np.allclose(sp, [5.26640187, 1.20553485, 3.69160487, 1.62827991, 4.26513502, 1.38600307]))
        self.assertTrue(np.allclose(r, 22300.86788001542, rtol=1e-8))
        self.assertFalse(t)
