import numpy as np
import unittest
from itertools import product
from pyrecest.distributions.se2_partially_wrapped_normal_distribution import SE2PartiallyWrappedNormalDistribution

from scipy.stats import multivariate_normal

class SE2PWNDistributionTest(unittest.TestCase):

    def setUp(self):
        self.mu = np.array([2, 3, 4])
        self.si1, self.si2, self.si3 = 0.9, 1.5, 1.7
        self.rho12, self.rho13, self.rho23 = 0.5, 0.3, 0.4
        self.C = np.array([
            [self.si1**2, self.si1*self.si2*self.rho12, self.si1*self.si3*self.rho13],
            [self.si1*self.si2*self.rho12, self.si2**2, self.si2*self.si3*self.rho23],
            [self.si1*self.si3*self.rho13, self.si2*self.si3*self.rho23, self.si3**2]
        ])
        self.pwn = SE2PartiallyWrappedNormalDistribution(self.mu, self.C)

    @staticmethod
    def _loop_wrapped_pdf(x, mu, C, n_wrappings=10):
        bound_dim = 1
        # Ensure x is at least 2D for iteration
        x = np.array(np.atleast_2d(x), dtype=np.float64)
        
        n_samples = x.shape[0]
        results = np.zeros(n_samples)
        
        # Generate all combinations of offsets for the bound_dim dimensions
        offset_values = [i*2*np.pi for i in range(-n_wrappings, n_wrappings+1)]
        all_combinations = list(product(offset_values, repeat=bound_dim))
        
        # Iterate over each sample
        for i in range(n_samples):
            sample = x[i]
            p = 0
            # Iterate over each offset combination and add to the sample before evaluating the PDF
            for offset in all_combinations:
                shifted_sample = sample.copy()
                shifted_sample[:bound_dim] += np.array(offset)
                p += multivariate_normal.pdf(shifted_sample, mu, C)
            results[i] = p
            
        # If input was 1D, return a single value; otherwise, return the array
        return results[0] if x.shape[0] == 1 else results

    def test_pdf(self):
        self.assertAlmostEqual(self.pwn.pdf(self.mu), SE2PWNDistributionTest._loop_wrapped_pdf(self.mu, self.mu, self.C), places=10)
        self.assertAlmostEqual(self.pwn.pdf(self.mu-1), SE2PWNDistributionTest._loop_wrapped_pdf(self.mu-1, self.mu, self.C), places=10)
        self.assertAlmostEqual(self.pwn.pdf(self.mu+2), SE2PWNDistributionTest._loop_wrapped_pdf(self.mu+2, self.mu, self.C), places=10)
        x = np.random.rand(20, 3)
        np.testing.assert_allclose(self.pwn.pdf(x), SE2PWNDistributionTest._loop_wrapped_pdf(x, self.mu, self.C, n_wrappings=10), rtol=1e-10)

    def test_pdf_large_uncertainty(self):
        C_high = 100 * np.eye(3, 3)
        pwn_large_uncertainty = SE2PartiallyWrappedNormalDistribution(self.mu, C_high)
        for t in range(1, 7):
            # Verify they are equal for 3 wrappings (same number of wrappings as in the class)
            pdf_class = pwn_large_uncertainty.pdf(self.mu + np.array([t, 0, 0]))
            np.testing.assert_allclose(pdf_class,
                                       SE2PWNDistributionTest._loop_wrapped_pdf(self.mu + np.array([t, 0, 0]), self.mu, C_high, n_wrappings=3),
                                       rtol=0.00001)
        
            # Verify they are unequal for 10 wrappings when the covariance is high
            pdf_loop_nested_10 = SE2PWNDistributionTest._loop_wrapped_pdf(self.mu + np.array([t, 0, 0]), self.mu, C_high, n_wrappings=10)

            # Calculate the relative errors
            relative_errors = np.abs(pdf_class - pdf_loop_nested_10) / pdf_class
            # Find the maximum relative error
            max_relative_error = np.max(relative_errors)
            self.assertGreater(max_relative_error, 0.00001)

    def test_integral(self):
        self.assertAlmostEqual(self.pwn.integrate(), 1, places=5)

    def test_sampling(self):
        np.random.seed(0)
        n = 10
        s = self.pwn.sample(n)
        self.assertEqual(s.shape[0], n)
        self.assertEqual(s.shape[1], 3)
        s = s[:, 0]
        self.assertTrue(np.all(s >= 0))
        self.assertTrue(np.all(s < 2 * np.pi))


if __name__ == '__main__':
    unittest.main()
