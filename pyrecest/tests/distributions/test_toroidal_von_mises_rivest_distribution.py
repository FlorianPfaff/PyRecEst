import unittest
import numpy as np
from pyrecest.distributions.hypertorus.toroidal_von_mises_rivest_distribution import ToroidalVonMisesRivestDistribution

class ToroidalVMRivestDistributionTest(unittest.TestCase):
    
    def setUp(self):
        self.mu = np.array([1, 2])
        self.kappa = np.array([0.7, 1.4])
        self.alpha = 0.3
        self.beta = 0.5
        self.tvm = ToroidalVonMisesRivestDistribution(self.mu, self.kappa, self.alpha, self.beta)
    
    def test_sanity_check(self):
        self.assertIsInstance(self.tvm, ToroidalVonMisesRivestDistribution)
        np.testing.assert_array_equal(self.tvm.mu, self.mu)
        np.testing.assert_array_equal(self.tvm.kappa, self.kappa)
        self.assertEqual(self.tvm.alpha, self.alpha)
        self.assertEqual(self.tvm.beta, self.beta)

    def test_integral(self):
        # Assuming the `integral` and `trigonometricMomentNumerical` methods are defined in your main class
        self.assertAlmostEqual(self.tvm.integral(), 1, delta=1E-5)
        np.testing.assert_array_almost_equal(self.tvm.trigonometricMomentNumerical(0), np.array([[1], [1]]), decimal=5)

    def test_shift(self):
        tvm = ToroidalVonMisesRivestDistribution(np.array([[3], [5]]), np.array([[0.7], [1.3]]), 0.4, 0.3)
        s = np.array([4, 2])
        tvm2 = tvm.shift(s)
        self.assertIsInstance(tvm2, ToroidalVonMisesRivestDistribution)

if __name__ == "__main__":
    unittest.main()
