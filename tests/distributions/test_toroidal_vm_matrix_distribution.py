import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, mod, pi
from pyrecest.distributions.hypertorus.toroidal_vm_matrix_distribution import (
    ToroidalVMMatrixDistribution,
)


class TestToroidalVMMatrixDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.5, 0.7])
        self.A = array([[0.3, 0.1], [-0.2, 0.4]])
        self.tvm = ToroidalVMMatrixDistribution(self.mu, self.kappa, self.A)

    def test_instance(self):
        self.assertIsInstance(self.tvm, ToroidalVMMatrixDistribution)

    def test_properties(self):
        npt.assert_allclose(self.tvm.mu, self.mu, atol=1e-10)
        npt.assert_allclose(self.tvm.kappa, self.kappa, atol=1e-10)
        npt.assert_allclose(self.tvm.A, self.A, atol=1e-10)

    def test_constructor_accepts_list_inputs(self):
        tvm = ToroidalVMMatrixDistribution(
            [1.0, 2.0], [0.5, 0.7], [[0.3, 0.1], [-0.2, 0.4]]
        )
        npt.assert_allclose(tvm.mu, self.mu, atol=1e-10)
        npt.assert_allclose(tvm.kappa, self.kappa, atol=1e-10)
        npt.assert_allclose(tvm.A, self.A, atol=1e-10)

    def test_constructor_rejects_invalid_parameters(self):
        invalid_cases = [
            ([1.0], self.kappa, self.A, "mu"),
            (self.mu, [0.5], self.A, "kappa"),
            (self.mu, [0.0, 0.7], self.A, "positive"),
            (self.mu, [-0.5, 0.7], self.A, "positive"),
            (self.mu, [float("nan"), 0.7], self.A, "finite"),
            (self.mu, self.kappa, [[0.3, 0.1]], "shape"),
            (self.mu, self.kappa, [[float("nan"), 0.1], [-0.2, 0.4]], "finite"),
        ]

        for mu, kappa, A, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ToroidalVMMatrixDistribution(mu, kappa, A)

    def test_pdf_positive(self):
        xs = array([[0.5, 1.0], [1.0, 2.0], [3.0, 4.0]])
        vals = self.tvm.pdf(xs)
        for v in vals.ravel():
            self.assertGreater(float(v), 0.0)

    def test_pdf_accepts_list_inputs(self):
        array_vals = self.tvm.pdf(array([[0.5, 1.0], [1.0, 2.0]]))
        list_vals = self.tvm.pdf([[0.5, 1.0], [1.0, 2.0]])
        npt.assert_allclose(list_vals, array_vals, atol=1e-10)

        single_val = self.tvm.pdf([1.0, 2.0])
        self.assertEqual(single_val.shape, (1,))
        npt.assert_allclose(single_val, self.tvm.pdf(array([[1.0, 2.0]])), atol=1e-10)

    def test_pdf_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            self.tvm.pdf([1.0, 2.0, 3.0])

    def test_mu_wrapped(self):
        mu_unwrapped = array([1.0 + 2 * float(pi), 2.0])
        tvm2 = ToroidalVMMatrixDistribution(mu_unwrapped, self.kappa, self.A)
        npt.assert_allclose(tvm2.mu, self.tvm.mu, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.tvm.integrate(), 1.0, delta=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral_numerical_normalization(self):
        # High concentration forces numerical normalization
        tvm_high = ToroidalVMMatrixDistribution(
            array([0.5, 1.0]), array([2.0, 2.0]), array([[0.1, 0.0], [0.0, 0.1]])
        )
        self.assertAlmostEqual(tvm_high.integrate(), 1.0, delta=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_multiply_integrates_to_1(self):
        tvm2 = ToroidalVMMatrixDistribution(
            array([0.3, 1.5]), array([0.4, 0.6]), array([[0.1, 0.0], [0.0, 0.2]])
        )
        product = self.tvm.multiply(tvm2)
        self.assertIsInstance(product, ToroidalVMMatrixDistribution)
        self.assertAlmostEqual(product.integrate(), 1.0, delta=2e-3)

    def test_multiply_rejects_wrong_type(self):
        with self.assertRaisesRegex(ValueError, "ToroidalVMMatrixDistribution"):
            self.tvm.multiply(object())

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_marginalize_to_1d_dim0(self):
        marginal = self.tvm.marginalize_to_1d(0)
        self.assertAlmostEqual(marginal.integrate(), 1.0, delta=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_marginalize_to_1d_dim1(self):
        marginal = self.tvm.marginalize_to_1d(1)
        self.assertAlmostEqual(marginal.integrate(), 1.0, delta=1e-4)

    def test_marginalize_to_1d_rejects_invalid_dimension(self):
        for dimension in (-1, 2):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(ValueError, "dimension"):
                    self.tvm.marginalize_to_1d(dimension)

    def test_shift(self):
        shift = array([0.5, -0.3])
        shifted = self.tvm.shift(shift)
        expected_mu = mod(self.mu + shift, 2.0 * float(pi))
        npt.assert_allclose(shifted.mu, expected_mu, atol=1e-10)
        # A and kappa unchanged
        npt.assert_allclose(shifted.kappa, self.tvm.kappa, atol=1e-10)
        npt.assert_allclose(shifted.A, self.tvm.A, atol=1e-10)

    def test_shift_accepts_list_input(self):
        shifted = self.tvm.shift([0.5, -0.3])
        expected_mu = mod(self.mu + array([0.5, -0.3]), 2.0 * float(pi))
        npt.assert_allclose(shifted.mu, expected_mu, atol=1e-10)

    def test_shift_does_not_modify_original(self):
        original_mu = array(self.tvm.mu)
        _ = self.tvm.shift(array([1.0, 1.0]))
        npt.assert_allclose(self.tvm.mu, original_mu, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="JAX arrays are immutable",
    )
    def test_shift_returns_independent_mutable_parameters(self):
        original_kappa = copy.deepcopy(self.tvm.kappa)
        original_A = copy.deepcopy(self.tvm.A)

        shifted = self.tvm.shift([0.5, -0.3])
        shifted.kappa[0] = shifted.kappa[0] + 1.0
        shifted.A[0, 0] = shifted.A[0, 0] + 1.0

        npt.assert_allclose(self.tvm.kappa, original_kappa, atol=1e-10)
        npt.assert_allclose(self.tvm.A, original_A, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
