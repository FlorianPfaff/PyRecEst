import numpy as np
import numpy.testing as npt
from pyrecest.distributions.hypertorus.fejer import (
    fejer_reduce_coefficients,
    fejer_weights,
)


def _evaluate_centered_1d(coefficients, xs):
    coefficients = np.asarray(coefficients)
    order = (coefficients.size - 1) // 2
    ks = np.arange(-order, order + 1)
    return np.sum(
        coefficients[:, None] * np.exp(1j * ks[:, None] * xs[None, :]), axis=0
    ).real


def test_fejer_reduction_preserves_zero_frequency_coefficient():
    coeff = np.array([1.0 + 2.0j, 2.0, 3.0, 2.0, 1.0 - 2.0j])
    reduced = fejer_reduce_coefficients(coeff, (3,))

    assert reduced.shape == (3,)
    assert reduced[1] == coeff[2]


def test_fejer_reduction_matches_tensor_product_weights_after_center_crop():
    coeff = np.ones((5, 7), dtype=complex)
    reduced = fejer_reduce_coefficients(coeff, (3, 5))

    npt.assert_allclose(reduced, fejer_weights((3, 5)))


def test_fejer_reduction_of_nonnegative_product_has_no_negative_grid_values():
    # f(x) = (1 + cos(x)) / (2*pi) is nonnegative and has order one.
    coeff = np.array([1.0 / (4.0 * np.pi), 1.0 / (2.0 * np.pi), 1.0 / (4.0 * np.pi)])

    # The product f(x)^2 has order two. Reducing it back to order one by a
    # Fejer mean should remain nonnegative because the Fejer kernel is positive.
    product_coeff = np.convolve(coeff, coeff, mode="full")
    reduced = fejer_reduce_coefficients(product_coeff, (3,))

    xs = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    vals = _evaluate_centered_1d(reduced, xs)
    assert vals.min() >= -1e-12
