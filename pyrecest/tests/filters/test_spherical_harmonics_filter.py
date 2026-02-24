import unittest
import numpy as np
from scipy.stats import norm

from pyrecest.distributions import VonMisesFisherDistribution, SphericalHarmonicsDistributionComplex
from pyrecest.filters import VonMisesFisherFilter
from pyrecest.filters.spherical_harmonics_filter import SphericalHarmonicsFilter

class SphericalHarmonicsFilterTest(unittest.TestCase):

    def test_update_identity(self):
        for transformation in ['identity', 'sqrt']:
            shd_filter = SphericalHarmonicsFilter(30, transformation)
            vmf_filter = VonMisesFisherFilter()

            vmf1 = VonMisesFisherDistribution(np.array([0, 1, 0]), 1)
            vmf2 = VonMisesFisherDistribution(np.array([0, 0, 1]), 0.1)
            
            shd1 = SphericalHarmonicsDistributionComplex.from_distribution_numerical_fast(vmf1, 30, transformation)
            shd2 = SphericalHarmonicsDistributionComplex.from_distribution_numerical_fast(vmf2, 30, transformation)

            vmf_filter.set_state(vmf1)
            vmf_filter.update_identity(vmf2, np.array([1, 0, 0]))

            shd_filter.set_state(shd1)
            shd_filter.update_identity(shd2, np.array([1, 0, 0]))

            self.assertTrue(np.allclose(vmf_filter.get_estimate_mean(), shd_filter.get_estimate_mean(), atol=1e-10))

    def test_update_using_likelihood(self):
        np.random.seed(1)

        pos_true = -1 / np.sqrt(3) * np.ones(3)

        # Generate measurements according to truncated gaussian along x, y, and z axis
        sigma_x = 0.3
        sigma_y = 0.3
        sigma_z = 0.3
        
        meas_x = self._generate_truncated_normals(pos_true[0], sigma_x, 5)
        meas_y = self._generate_truncated_normals(pos_true[1], sigma_y, 5)
        meas_z = self._generate_truncated_normals(pos_true[2], sigma_z, 5)

        for transformation in ['identity', 'sqrt']:
            sh_filter = SphericalHarmonicsFilter(11, transformation)

            for x in meas_x:
                sh_filter.update_nonlinear(lambda z, x: norm.pdf(z[0], x[0], sigma_x), np.array([x, 0, 0]))
            for y in meas_y:
                sh_filter.update_nonlinear(lambda z, x: norm.pdf(z[1], x[1], sigma_y), np.array([0, y, 0]))
            for z in meas_z:
                sh_filter.update_nonlinear(lambda z, x: norm.pdf(z[2], x[2], sigma_z), np.array([0, 0, z]))
            
            self.assertTrue(np.allclose(sh_filter.get_estimate_mean(), pos_true, atol=0.3))

    def test_update_using_likelihood_multiple(self):
        for transformation in ['identity', 'sqrt']:
            sh_filter1 = SphericalHarmonicsFilter(10, transformation)
            sh_filter2 = SphericalHarmonicsFilter(10, transformation)

            sigma_x = 0.3
            sigma_y = 0.3
            sigma_z = 0.3
            
            sh_filter1.update_nonlinear(lambda z, x: norm.pdf(z[0], x[0], sigma_x), np.array([-1 / np.sqrt(3), 0, 0]))
            sh_filter1.update_nonlinear(lambda z, x: norm.pdf(z[1], x[1], sigma_y), np.array([0, -1 / np.sqrt(3), 0]))
            sh_filter1.update_nonlinear(lambda z, x: norm.pdf(z[2], x[2], sigma_z), np.array([0, 0, -1 / np.sqrt(3)]))

            sh_filter2.update_nonlinear_multiple(
                [
                    lambda z, x: norm.pdf(z[0], x[0], sigma_x),
                    lambda z, x: norm.pdf(z[1], x[1], sigma_y),
                    lambda z, x: norm.pdf(z[2], x[2], sigma_z)
                ],
                [
                    np.array([-1 / np.sqrt(3), 0, 0]),
                    np.array([0, -1 / np.sqrt(3), 0]),
                    np.array([0, 0, -1 / np.sqrt(3)])
                ]
            )

            self.assertTrue(np.allclose(sh_filter2.get_estimate_mean(), sh_filter1.get_estimate_mean(), atol=1e-5))

    def test_prediction_sqrt_vs_id(self):
        degree = 21
        density_init = VonMisesFisherDistribution(np.array([1, 1, 0]) / np.sqrt(2), 2)
        sys_noise = VonMisesFisherDistribution(np.array([0, 0, 1]), 1)

        shd_init_id = SphericalHarmonicsDistributionComplex.from_distribution_numerical_fast(density_init, degree, 'identity')
        shd_init_sqrt = SphericalHarmonicsDistributionComplex.from_distribution_numerical_fast(density_init, degree, 'sqrt')
        shd_noise_id = SphericalHarmonicsDistributionComplex.from_distribution_numerical_fast(sys_noise, degree, 'identity')
        shd_noise_sqrt = SphericalHarmonicsDistributionComplex.from_distribution_numerical_fast(sys_noise, degree, 'sqrt')

        sh_filter1 = SphericalHarmonicsFilter(degree, 'identity')
        sh_filter2 = SphericalHarmonicsFilter(degree, 'sqrt')

        sh_filter1.set_state(shd_init_id)
        sh_filter2.set_state(shd_init_sqrt)

        sh_filter1.predict_identity(shd_noise_id)
        sh_filter2.predict_identity(shd_noise_sqrt)

        np.testing.assert_allclose(sh_filter1.state.total_variation_distance_numerical(sh_filter2.state), 0, atol=5e-15)

    # Helper function to generate truncated normals
    def _generate_truncated_normals(self, mu, sigma, n):
        samples = []
        while len(samples) < n:
            sample = np.random.normal(mu, sigma)
            if -1 <= sample <= 1:
                samples.append(sample)
        return samples
