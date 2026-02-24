import unittest
import numpy as np
from pyrecest.distributions.hypertorus.toroidal_fourier_distribution import ToroidalFourierDistribution
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import ToroidalVonMisesSineDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution


class ToroidalFourierDistributionTest(unittest.TestCase):
    def setUp(self):
        self.tvm = ToroidalVonMisesSineDistribution(np.array([1, 4]), np.array([0.3, 0.7]), 0.5)
        self.tfd = ToroidalFourierDistribution.from_distribution(self.tvm, [15, 15], 'sqrt')
        self.hfd = HypertoroidalFourierDistribution.from_distribution(self.tvm, [15, 15], 'sqrt')
        self.test_points = np.random.rand(100, 2) * 4 * np.pi - np.pi

    def test_single_element_vector(self):
        ToroidalFourierDistribution([1])

    def test_from_function(self):
        tvm = ToroidalVonMisesSineDistribution(np.array([1, 2]), np.array([0.3, 0.5]), 0.5)
        tfd1_fun = ToroidalFourierDistribution.from_function(
            lambda x, y: np.reshape(tvm.pdf(np.column_stack([x.ravel(), y.ravel()])), x.shape),
            [15, 15],
            desired_transformation='identity'
        )
        tfd2_fun = ToroidalFourierDistribution.from_function(
            lambda x, y: np.reshape(tvm.pdf(np.column_stack([x.ravel(), y.ravel()])), x.shape),
            [15, 15],
            desired_transformation='sqrt'
        )
        for i in range(-2, 4):
            tvm_moment = tvm.trigonometric_moment(i)
            np.testing.assert_allclose(tfd1_fun.trigonometric_moment(i), tvm_moment, atol=1e-6)
            np.testing.assert_allclose(tfd1_fun.trigonometric_moment_numerical(i), tvm_moment, atol=1e-6)
            np.testing.assert_allclose(tfd2_fun.trigonometric_moment(i), tvm_moment, atol=1e-6)
            np.testing.assert_allclose(tfd2_fun.trigonometric_moment_numerical(i), tvm_moment, atol=1e-6)

    def test_from_function_and_moments(self):
        tvm = ToroidalVonMisesSineDistribution([1, 2], [0.3, 0.5], 0.5)
        tfd1 = ToroidalFourierDistribution.from_distribution(tvm, [15, 15], 'identity')
        tfd2 = ToroidalFourierDistribution.from_distribution(tvm, [15, 15], 'sqrt')
        for i in range(-2, 4):
            tvm_moment = tvm.trigonometric_moment(i)
            np.testing.assert_allclose(tfd1.trigonometric_moment(i), tvm_moment, atol=1e-6)
            np.testing.assert_allclose(tfd1.trigonometric_moment_numerical(i), tvm_moment, atol=1e-6)
            np.testing.assert_allclose(tfd2.trigonometric_moment(i), tvm_moment, atol=1e-6)
            np.testing.assert_allclose(tfd2.trigonometric_moment_numerical(i), tvm_moment, atol=1e-6)

        with np.testing.assert_warns(Warning):
            np.testing.assert_allclose(tfd1.trigonometric_moment(16), [0, 0])
            np.testing.assert_allclose(tfd2.trigonometric_moment(30), [0, 0])

    def test_integrate(self):
        tvm = ToroidalVonMisesSineDistribution([1, 2], [0.6, 1], 0.2)
        tfd_id = ToroidalFourierDistribution.from_distribution(tvm, [45, 49], 'identity')
        tfd_sqrt = ToroidalFourierDistribution.from_distribution(tvm, [45, 49], 'sqrt')
        l = [0.3, 0.3]
        r = [1.5, 1.5]
        np.testing.assert_allclose(tfd_id.integrate(l, r), tvm.integrate(l, r), atol=1e-6)
        np.testing.assert_allclose(tfd_id.integral_numerical(l, r), tvm.integrate(l, r), atol=1e-6)
        np.testing.assert_allclose(tfd_sqrt.integrate(l, r), tvm.integrate(l, r), atol=1e-6)
        np.testing.assert_allclose(tfd_sqrt.integral_numerical(l, r), tvm.integrate(l, r), atol=1e-6)

        tfd_simple = ToroidalFourierDistribution(np.diag([0, 1 / (4 * np.pi ** 2), 0]), 'identity')
        np.testing.assert_allclose(tfd_simple.integrate(), 1, atol=1e-6)

    def test_to_twn(self):
        mu = [1, 3]
        C = [[9, 0.3], [0.3, 2]]
        twn = HypertoroidalWrappedNormalDistribution(mu, C)
        tfd = ToroidalFourierDistribution.from_distribution(twn, 9, 'identity')
        twn_conv = tfd.to_twn()
        np.testing.assert_allclose(twn_conv.mu, twn.mu, atol=1e-4)
        np.testing.assert_allclose(twn_conv.C, twn.C, atol=1e-4)
        tfd = ToroidalFourierDistribution.from_distribution(twn, 9, 'sqrt')
        twn_conv = tfd.to_twn()
        np.testing.assert_allclose(twn_conv.mu, twn.mu, atol=1e-4)
        np.testing.assert_allclose(twn_conv.C, twn.C, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
