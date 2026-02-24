import unittest
import numpy as np
from scipy import linalg
from numpy.random import default_rng
from pyrecest.distributions.hypertorus.toroidal_von_mises_matrix_distribution import ToroidalVonMisesMatrixDistribution
from pyrecest.distributions.hypertorus.toroidal_fourier_distribution import ToroidalFourierDistribution

class ToroidalVMMatrixDistributionTest(unittest.TestCase):

    def setUp(self):
        self.mu = np.array([1.7, 2.3])
        self.kappa = np.array([0.8, 0.3])
        self.A = 0.1 * np.array([[1, 2], [3, 4]])
        self.tvm = ToroidalVonMisesMatrixDistribution(self.mu, self.kappa, self.A)
        X, Y = np.meshgrid(range(1, 7), range(1, 7))
        self.testpoints = np.column_stack([X.ravel(), Y.ravel()])
        self.tvm2 = ToroidalVonMisesMatrixDistribution(self.mu + 1, self.kappa + 0.2, linalg.inv(self.A))


    def test_sanity_check(self):
        self.assertIsInstance(self.tvm, ToroidalVonMisesMatrixDistribution)
        self.assertTrue(np.array_equal(self.tvm.mu, self.mu))
        self.assertTrue(np.array_equal(self.tvm.kappa, self.kappa))
        self.assertTrue(np.array_equal(self.tvm.A, self.A))

    def test_pdf(self):
        def pdf(xs, mu, kappa, A, C):
            if xs.shape[1] > 1:
                p = np.zeros((1, xs.shape[1]))
                for i in range(xs.shape[1]):
                    p[0, i] = pdf(xs[:, i:i + 1], mu, kappa, A, C)
                return p
            return C * np.exp(
                kappa[0] * np.cos(xs[0, 0] - mu[0]) +
                kappa[1] * np.cos(xs[1, 0] - mu[1]) +
                np.array([np.cos(xs[0, 0] - mu[0]), np.sin(xs[0, 0] - mu[0])]) @
                A @
                np.array([np.cos(xs[1, 0] - mu[1]), np.sin(xs[1, 0] - mu[1])])
            )

        np.testing.assert_allclose(self.tvm.pdf(self.testpoints), np.squeeze(pdf(self.testpoints.T, self.mu, self.kappa, self.A, self.tvm.C)), rtol=1e-10)

    def test_integral(self):
        self.assertAlmostEqual(self.tvm.integrate(), 1, places=5)
        
    def test_trig_moment_numerical(self):
        np.testing.assert_allclose(self.tvm.trigonometric_moment_numerical(0), np.array([1, 1]))

    def test_multiply(self):
        tvmMul = self.tvm.multiply(self.tvm2)
        tvmMulSwapped = self.tvm2.multiply(self.tvm)

        C = self.tvm.pdf(np.array([[0], [0]])) * self.tvm2.pdf(np.array([[0], [0]])) / tvmMul.pdf(np.array([[0], [0]]))
        np.testing.assert_allclose(self.tvm.pdf(self.testpoints) * self.tvm2.pdf(self.testpoints), C * tvmMul.pdf(self.testpoints), rtol=1e-10)
        np.testing.assert_allclose(self.tvm.pdf(self.testpoints) * self.tvm2.pdf(self.testpoints), C * tvmMulSwapped.pdf(self.testpoints), rtol=1e-10)

    def test_compare_with_ToroidalFourier_multiplication(self):
        tvmMul = self.tvm.multiply(self.tvm2)
        tvmMulSwapped = self.tvm2.multiply(self.tvm)
        n = 45
        tf = ToroidalFourierDistribution.from_distribution(self.tvm, n)
        tf2 = ToroidalFourierDistribution.from_distribution(self.tvm2, n)
        tfMul = tf.multiply(tf2)
        tfMulSwapped = tf2.multiply(tf)

        tvmMul.C = 1
        tvmMul.C = 1 / tvmMul.integrate()
        tvmMulSwapped.C = 1
        tvmMulSwapped.C = 1 / tvmMulSwapped.integrate()

        np.testing.assert_allclose(tfMul.pdf(self.testpoints), tvmMul.pdf(self.testpoints), atol=1e-5)
        np.testing.assert_allclose(tfMulSwapped.pdf(self.testpoints), tvmMulSwapped.pdf(self.testpoints), atol=1e-5)

    def test_product_approximation(self):
        mu1 = np.array([1.7, 0.5])
        kappa1 = np.array([0.8, 0.3])
        A1 = 0.1 * np.array([[1, 0.2], [0.2, 1]])
        mu2 = np.array([1.5, 1])
        kappa2 = np.array([0.7, 0.2])
        A2 = 0.1 * np.array([[1, -0.2], [-0.2, 1]])

        tvm1 = ToroidalVonMisesMatrixDistribution(mu1, kappa1, A1)
        tvm2 = ToroidalVonMisesMatrixDistribution(mu2, kappa2, A2)
        tvm3 = tvm1.multiply(tvm2)

        rng = default_rng(1234)
        n = 1e4
        samples = rng.uniform(0, 2 * np.pi, (2, int(n)))
        weights1 = tvm1.pdf(samples)
        weights2 = tvm2.pdf(samples)
        weights = weights1 * weights2

        weights = weights / np.sum(weights)

        muApprox = np.average(samples, weights=weights, axis=1)
        kappaApprox = np.average(np.cos(samples - muApprox[:, np.newaxis]), weights=weights, axis=1)
        kappaApprox = -np.log(-kappaApprox)

        Aapprox = np.zeros((2, 2))
        Aapprox[0, 0] = np.average(np.cos(samples[0, :] - muApprox[0]) * np.cos(samples[1, :] - muApprox[1]), weights=weights)
        Aapprox[0, 1] = np.average(np.cos(samples[0, :] - muApprox[0]) * np.sin(samples[1, :] - muApprox[1]), weights=weights)
        Aapprox[1, 0] = np.average(np.sin(samples[0, :] - muApprox[0]) * np.cos(samples[1, :] - muApprox[1]), weights=weights)
        Aapprox[1, 1] = np.average(np.sin(samples[0, :] - muApprox[0]) * np.sin(samples[1, :] - muApprox[1]), weights=weights)

        tvmApprox = ToroidalVonMisesMatrixDistribution(muApprox, kappaApprox, Aapprox)

        np.testing.assert_allclose(tvmApprox.pdf(self.testpoints), tvm3.pdf(self.testpoints), atol=1e-2, rtol=1e-1)

if __name__ == "__main__":
    unittest.main()
