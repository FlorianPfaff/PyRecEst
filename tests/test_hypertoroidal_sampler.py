import unittest
from fractions import Fraction

import numpy as np
import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, diff, pi, std
from pyrecest.sampling.hypertoroidal_sampler import (
    CircularUniformSampler,
    _validate_integral_scalar,
)


class TestCircularUniformSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = CircularUniformSampler()

    def test_sample_stochastic(self):
        n_samples = 1000
        samples = self.sampler.sample_stochastic(n_samples)

        # Check that the returned shape matches the requested number of samples
        self.assertEqual(samples.shape[0], n_samples)

        # Check that all samples are within the range [0, 2*pi)
        self.assertTrue(all(samples >= 0.0))
        self.assertTrue(all(samples < 2.0 * pi))

    def test_sample_stochastic_accepts_integer_like_scalars(self):
        samples = self.sampler.sample_stochastic(np.array(3.0), dim=np.array(1.0))

        self.assertEqual(samples.shape, (3, 1))

    def test_sample_stochastic_rejects_invalid_controls(self):
        invalid_sample_counts = (-1, 1.5, True, "3", b"3", [2])
        for n_samples in invalid_sample_counts:
            with self.subTest(n_samples=n_samples):
                with self.assertRaises(ValueError):
                    self.sampler.sample_stochastic(n_samples)

        invalid_dims = (0, 1.5, 2, True, "1", b"1", [1])
        for dim in invalid_dims:
            with self.subTest(dim=dim):
                with self.assertRaises(ValueError):
                    self.sampler.sample_stochastic(2, dim=dim)

    def test_integral_controls_reject_fractional_values_rounded_by_binary64(self):
        rounded_half_integer = Fraction(2**54 + 1, 2)

        with self.assertRaisesRegex(ValueError, "must be a finite integer"):
            _validate_integral_scalar(rounded_half_integer, "value", minimum=0)

    def test_integral_controls_preserve_exact_large_integer(self):
        exact_integer = Fraction(2**54 + 2, 2)

        self.assertEqual(
            _validate_integral_scalar(exact_integer, "value", minimum=0),
            2**53 + 1,
        )

    def test_get_grid(self):
        grid_density_parameter = 100
        grid_points = self.sampler.get_grid(grid_density_parameter)

        # Check that the returned shape matches the grid density parameter
        self.assertEqual(grid_points.shape[0], grid_density_parameter)

        # Check that all grid points are within the range [0, 2*pi)
        self.assertTrue(all(grid_points >= 0.0))
        self.assertTrue(all(grid_points < 2.0 * pi))

        # Check that the grid points are equidistant
        npt.assert_array_almost_equal(std(diff(grid_points)), 0.0)

    def test_get_grid_rejects_nonpositive_density(self):
        with self.assertRaises(ValueError):
            self.sampler.get_grid(0)
        with self.assertRaises(ValueError):
            self.sampler.get_grid(-1)

    def test_get_grid_accepts_integer_like_scalar_density(self):
        grid_points = self.sampler.get_grid(np.array(4.0))

        self.assertEqual(grid_points.shape[0], 4)

    def test_get_grid_rejects_noninteger_density(self):
        for density in (2.5, True, "3", b"3", [3]):
            with self.subTest(density=density):
                with self.assertRaises(ValueError):
                    self.sampler.get_grid(density)


if __name__ == "__main__":
    unittest.main()
