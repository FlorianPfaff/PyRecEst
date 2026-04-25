import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    all,
    allclose,
    array,
    concatenate,
    exp,
    eye,
    isreal,
    linalg,
    ones,
    random,
    zeros,
)
from pyrecest.distributions.hypersphere_subset.bayesian_complex_watson_mixture_model import (
    BayesianComplexWatsonMixtureModel,
    _complex_bingham_first_order_moments,
    _simplex_integral,
)
from pyrecest.distributions.hypersphere_subset.complex_watson_distribution import (
    ComplexWatsonDistribution,
)


def _make_unit_vectors(D, N):
    Z = random.normal(size=(D, N)) + 1j * random.normal(size=(D, N))
    Z /= linalg.norm(Z, axis=0)
    return Z


class TestSimplexIntegralMixture(unittest.TestCase):
    def test_D2(self):
        a, b = 2.0, 1.0
        expected = float((exp(a) - exp(b)) / (a - b))
        self.assertAlmostEqual(
            float(_simplex_integral(array([a, b]))), expected, places=6
        )

    def test_D3(self):
        Lambda = array([2.0, 1.0, 0.0])
        expected = float(exp(2)) / 2 - float(exp(1)) + 0.5
        self.assertAlmostEqual(float(_simplex_integral(Lambda)), expected, places=5)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
class TestComplexBinghamFirstOrderMoments(unittest.TestCase):
    def test_uniform_zero_eigenvalues(self):
        D = 3
        Lambda = zeros(D)
        moments = _complex_bingham_first_order_moments(Lambda, D)
        npt.assert_allclose(moments, ones(D) / D, atol=1e-2)

    def test_sum_to_one(self):
        D = 4
        Lambda = array([0.0, -1.0, -2.0, -3.0])
        moments = _complex_bingham_first_order_moments(Lambda, D)
        self.assertAlmostEqual(float(moments.sum()), 1.0, places=5)

    def test_largest_eigenvalue_largest_moment(self):
        D = 3
        Lambda = array([0.0, -5.0, -10.0])
        moments = _complex_bingham_first_order_moments(Lambda, D)
        self.assertGreater(float(moments[0]), float(moments[1]))
        self.assertGreater(float(moments[1]), float(moments[2]))


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
class TestQuadraticExpectation(unittest.TestCase):
    def test_identity_input_equals_moments_sum(self):
        """E[z^H I z] = E[|z|^2] = 1 by definition."""
        D = 3
        K = 2
        B = zeros((D, D, K), dtype=complex)
        I_3d = eye(D, dtype=complex)[:, :, None]
        E = BayesianComplexWatsonMixtureModel.quadratic_expectation(I_3d, B)
        npt.assert_allclose(E, ones((1, K)), atol=1e-8)

    def test_shape(self):
        D = 4
        N = 10
        K = 3
        Z = _make_unit_vectors(D, N)
        dp = Z[:, None, :] * Z.conj()[None, :, :]
        dp = dp.reshape(D, D, N)
        B = zeros((D, D, K), dtype=complex)
        E = BayesianComplexWatsonMixtureModel.quadratic_expectation(dp, B)
        self.assertEqual(E.shape, (N, K))

    def test_real_output(self):
        D = 3
        N = 5
        K = 2
        Z = _make_unit_vectors(D, N)
        dp = (Z[:, None, :] * Z.conj()[None, :, :]).reshape(D, D, N)
        B = zeros((D, D, K), dtype=complex)
        E = BayesianComplexWatsonMixtureModel.quadratic_expectation(dp, B)
        self.assertTrue(all(isreal(E)))


class TestBayesianComplexWatsonMixtureModelConstructor(unittest.TestCase):
    def test_basic_construction(self):
        D, K = 3, 2
        B = zeros((D, D, K), dtype=complex)
        concentrations = array([5.0, 10.0])
        alpha = array([1.0, 1.0])
        model = BayesianComplexWatsonMixtureModel(B, concentrations, alpha)
        self.assertEqual(model.K, K)
        self.assertEqual(model.dim, D)

    def test_non_hermitian_B_raises(self):
        D, K = 2, 1
        B = ones((D, D, K), dtype=complex) * (1 + 1j)
        with self.assertRaises(AssertionError):
            BayesianComplexWatsonMixtureModel(B, array([1.0]), array([1.0]))


class TestParametersDefault(unittest.TestCase):
    def test_keys_present(self):
        params = BayesianComplexWatsonMixtureModel.parameters_default(4, 3)
        self.assertIn("initial", params)
        self.assertIn("prior", params)
        self.assertIn("I", params)
        self.assertIn("B", params["initial"])
        self.assertIn("kappa", params["initial"])
        self.assertIn("alpha", params["initial"])

    def test_shapes(self):
        D, K = 5, 4
        params = BayesianComplexWatsonMixtureModel.parameters_default(D, K)
        self.assertEqual(params["initial"]["B"].shape, (D, D, K))
        self.assertEqual(len(params["initial"]["alpha"]), K)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
class TestFitDefault(unittest.TestCase):
    def test_fit_returns_model(self):
        D, K, N = 3, 2, 50
        Z = _make_unit_vectors(D, N)
        model, _ = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        self.assertIsInstance(model, BayesianComplexWatsonMixtureModel)
        self.assertEqual(model.K, K)
        self.assertEqual(model.dim, D)

    def test_posterior_keys(self):
        D, K, N = 3, 2, 30
        Z = _make_unit_vectors(D, N)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        for key in ("B", "kappa", "alpha", "gamma"):
            self.assertIn(key, posterior)

    def test_gamma_sums_to_one(self):
        D, K, N = 3, 2, 40
        Z = _make_unit_vectors(D, N)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        gamma = posterior["gamma"]
        npt.assert_allclose(gamma.sum(axis=1), ones(N), atol=1e-8)

    def test_alpha_positive(self):
        D, K, N = 3, 2, 40
        Z = _make_unit_vectors(D, N)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        self.assertTrue(all(posterior["alpha"] > 0))

    def test_kappa_nonnegative(self):
        D, K, N = 3, 3, 60
        Z = _make_unit_vectors(D, N)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        self.assertTrue(all(posterior["kappa"] >= 0))

    def test_B_hermitian_after_fit(self):
        D, K, N = 3, 2, 40
        Z = _make_unit_vectors(D, N)
        model, _ = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        for k in range(K):
            self.assertTrue(
                allclose(model.B[:, :, k], model.B[:, :, k].conj().T, atol=1e-8),
                msg=f"B[:,:,{k}] is not Hermitian",
            )

    def test_fit_two_cluster_recovery(self):
        """Fit on data from two distinct clusters should assign high weight to each."""
        D = 3
        # Two orthogonal modes
        mu1 = array([1.0, 0.0, 0.0], dtype=complex)
        mu2 = array([0.0, 1.0, 0.0], dtype=complex)
        dist1 = ComplexWatsonDistribution(mu1, 20.0)
        dist2 = ComplexWatsonDistribution(mu2, 20.0)
        Z = concatenate([dist1.sample(60).T, dist2.sample(60).T], axis=1)
        K = 2
        params = BayesianComplexWatsonMixtureModel.parameters_default(D, K)
        params["I"] = 50
        _, posterior = BayesianComplexWatsonMixtureModel.fit(Z, params)
        # Both components should have non-trivial assignment
        N_k = posterior["gamma"].sum(axis=0)
        self.assertGreater(float(min(N_k)), 10.0)


if __name__ == "__main__":
    unittest.main()
