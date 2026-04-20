import numpy as _np
import numpy.testing as npt
import unittest
from itertools import product

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions.se2_partially_wrapped_normal_distribution import SE2PartiallyWrappedNormalDistribution

from scipy.stats import multivariate_normal

class SE2PWNDistributionTest(unittest.TestCase):

    def setUp(self):
        self.mu = array([2.0, 3.0, 4.0])
        si1, si2, si3 = 0.9, 1.5, 1.7
        rho12, rho13, rho23 = 0.5, 0.3, 0.4
        self.C = array([
            [si1**2, si1*si2*rho12, si1*si3*rho13],
            [si1*si2*rho12, si2**2, si2*si3*rho23],
            [si1*si3*rho13, si2*si3*rho23, si3**2]
        ])
        self.pwn = SE2PartiallyWrappedNormalDistribution(self.mu, self.C)

    @staticmethod
    def _loop_wrapped_pdf(x, mu, C, n_wrappings=10, bound_dim=1):
        # Ensure x is at least 2D for iteration; convert to plain numpy for scipy
        x = _np.array(_np.atleast_2d(x), dtype=_np.float64)
        mu = _np.asarray(mu, dtype=_np.float64)
        C = _np.asarray(C, dtype=_np.float64)
        # Wrap the periodic dimensions into [0, 2*pi) to match the class's pdf behavior
        x = x.copy()
        x[:, :bound_dim] = x[:, :bound_dim] % (2 * _np.pi)

        n_samples = x.shape[0]
        results = _np.zeros(n_samples)

        # Generate all combinations of offsets for the bound_dim dimensions
        offset_values = [i*2*_np.pi for i in range(-n_wrappings, n_wrappings+1)]
        all_combinations = list(product(offset_values, repeat=bound_dim))

        # Iterate over each sample
        for i in range(n_samples):
            sample = x[i]
            p = 0
            # Iterate over each offset combination and add to the sample before evaluating the PDF
            for offset in all_combinations:
                shifted_sample = sample.copy()
                shifted_sample[:bound_dim] += _np.array(offset)
                p += multivariate_normal.pdf(shifted_sample, mu, C)
            results[i] = p

        # If input was 1D, return a single value; otherwise, return the array
        return results[0] if x.shape[0] == 1 else results

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="scipy float64 comparison not meaningful with JAX float32",
    )
    def test_pdf(self):
        npt.assert_allclose(float(self.pwn.pdf(self.mu)[0]), SE2PWNDistributionTest._loop_wrapped_pdf(self.mu, self.mu, self.C), rtol=1e-5)
        npt.assert_allclose(float(self.pwn.pdf(self.mu - array([1.0, 1.0, 1.0]))[0]), SE2PWNDistributionTest._loop_wrapped_pdf(self.mu - array([1.0, 1.0, 1.0]), self.mu, self.C), rtol=1e-5)
        npt.assert_allclose(float(self.pwn.pdf(self.mu + array([2.0, 2.0, 2.0]))[0]), SE2PWNDistributionTest._loop_wrapped_pdf(self.mu + array([2.0, 2.0, 2.0]), self.mu, self.C), rtol=1e-5)
        x = array(_np.random.rand(20, 3))
        npt.assert_allclose(
            _np.asarray(self.pwn.pdf(x)),
            SE2PWNDistributionTest._loop_wrapped_pdf(x, self.mu, self.C, n_wrappings=10),
            rtol=1e-5,
        )

    def test_pdf_large_uncertainty(self):
        C_high = array(100 * _np.eye(3, 3))
        pwn_large_uncertainty = SE2PartiallyWrappedNormalDistribution(self.mu, C_high)
        for t in range(1, 7):
            # Verify they are equal for 3 wrappings (same number of wrappings as in the class)
            offset = array([float(t), 0.0, 0.0])
            pdf_class = _np.asarray(pwn_large_uncertainty.pdf(self.mu + offset))
            npt.assert_allclose(pdf_class,
                                       SE2PWNDistributionTest._loop_wrapped_pdf(self.mu + offset, self.mu, C_high, n_wrappings=3),
                                       rtol=0.00001)
        
            # Verify they are unequal for 10 wrappings when the covariance is high
            pdf_loop_nested_10 = SE2PWNDistributionTest._loop_wrapped_pdf(self.mu + offset, self.mu, C_high, n_wrappings=10)

            # Calculate the relative errors
            relative_errors = _np.abs(pdf_class - pdf_loop_nested_10) / pdf_class
            # Find the maximum relative error
            max_relative_error = _np.max(relative_errors)
            self.assertGreater(max_relative_error, 0.00001)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported for JAX backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.pwn.integrate(), 1, places=5)

    def test_sampling(self):
        _np.random.seed(0)
        n = 10
        s = self.pwn.sample(n)
        self.assertEqual(s.shape[0], n)
        self.assertEqual(s.shape[1], 3)
        s = _np.asarray(s[:, 0])
        self.assertTrue(_np.all(s >= 0))
        self.assertTrue(_np.all(s < 2 * _np.pi))

    def test_mean_4d_shape(self):
        m = self.pwn.mean_4d()
        self.assertEqual(m.shape, (4,))

    def test_mean_4d_values(self):
        mu0 = float(self.mu[0])
        c00 = float(self.C[0, 0])
        expected = _np.array(
            [
                _np.cos(mu0) * _np.exp(-c00 / 2),
                _np.sin(mu0) * _np.exp(-c00 / 2),
                float(self.mu[1]),
                float(self.mu[2]),
            ]
        )
        npt.assert_allclose(_np.asarray(self.pwn.mean_4d()), expected, rtol=1e-5)

    def test_covariance_4d_shape(self):
        cov = self.pwn.covariance_4d()
        self.assertEqual(cov.shape, (4, 4))

    def test_covariance_4d_symmetric(self):
        cov = _np.asarray(self.pwn.covariance_4d())
        npt.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_4d_matches_numerical(self):
        """Analytical covariance should match a numerical estimate within tolerance."""
        _np.random.seed(0)
        cov_analytical = _np.asarray(self.pwn.covariance_4d())
        cov_numerical = _np.asarray(self.pwn.covariance_4d_numerical(n_samples=100000))
        npt.assert_allclose(cov_analytical, cov_numerical, atol=0.2)

    def test_from_samples_recovers_params(self):
        """from_samples should recover mu and C (up to Monte-Carlo noise)."""
        _np.random.seed(42)
        samples = array(_np.asarray(self.pwn.sample(50000)))
        fitted = SE2PartiallyWrappedNormalDistribution.from_samples(samples)
        npt.assert_allclose(_np.asarray(fitted.mu), _np.asarray(self.mu), atol=0.05)
        npt.assert_allclose(_np.asarray(fitted.C), _np.asarray(self.C), atol=0.1)


if __name__ == '__main__':
    unittest.main()
