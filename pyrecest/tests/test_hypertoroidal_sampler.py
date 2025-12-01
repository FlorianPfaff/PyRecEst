import unittest

import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, diff, pi, std
from pyrecest.sampling.hypertoroidal_sampler import CircularUniformSampler


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


if __name__ == "__main__":
    unittest.main()
