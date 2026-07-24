import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.distributions.hypertorus.fejer import (
    adaptive_kernel_reduce_coefficients,
    apply_fejer_weights,
    centered_coefficients,
    coefficient_grid_shape,
    fejer_weights,
    korovkin_weights,
    minimum_on_fft_grid,
    reduce_coefficients,
)


def test_fejer_weights_1d():
    npt.assert_allclose(
        fejer_weights((5,)), np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])
    )


def test_fejer_weights_product():
    w = fejer_weights((3, 5))
    expected = np.outer(
        np.array([0.5, 1.0, 0.5]), np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])
    )
    npt.assert_allclose(w, expected)


def test_korovkin_first_multiplier_is_cosine_bound():
    order = 4
    weights = korovkin_weights((2 * order + 1,))
    center = order

    assert weights[center] == 1.0
    npt.assert_allclose(weights[center + 1], np.cos(np.pi / (order + 2)))
    npt.assert_allclose(weights[center - 1], np.cos(np.pi / (order + 2)))


def test_korovkin_weights_are_tensor_product():
    w = korovkin_weights((3, 5))
    expected = np.outer(korovkin_weights((3,)), korovkin_weights((5,)))
    npt.assert_allclose(w, expected)


def test_apply_fejer_weights_preserves_center():
    c = np.ones((5,), dtype=complex)
    weighted = apply_fejer_weights(c)
    assert weighted[2] == 1.0
    assert abs(weighted[0]) < abs(weighted[2])


def test_centered_coefficients_crop_and_pad():
    c = np.arange(7)
    npt.assert_array_equal(centered_coefficients(c, (3,)), np.array([2, 3, 4]))
    npt.assert_array_equal(
        centered_coefficients(c, (9,)), np.array([0, 0, 1, 2, 3, 4, 5, 6, 0])
    )


def test_reduce_coefficients_sharp_matches_centered_coefficients():
    c = np.arange(7)
    npt.assert_array_equal(
        reduce_coefficients(c, (3,), kernel="sharp"), centered_coefficients(c, (3,))
    )


@pytest.mark.parametrize("invalid_value", [1.5, np.float64(2.0), "2", True])
def test_coefficient_grid_shape_rejects_noninteger_oversampling_factor(invalid_value):
    with pytest.raises(
        ValueError, match="oversampling_factor must be a positive integer"
    ):
        coefficient_grid_shape((3,), oversampling_factor=invalid_value)


def test_coefficient_grid_shape_accepts_numpy_integer_oversampling_factor():
    assert coefficient_grid_shape((3,), oversampling_factor=np.int64(2)) == (5,)


def test_adaptive_reduction_keeps_nonnegative_sharp_result_unchanged():
    coeff = np.array([0.05, 1.0 / (2.0 * np.pi), 0.05])
    reduced, exponent = adaptive_kernel_reduce_coefficients(
        coeff, (3,), return_exponent=True
    )

    npt.assert_allclose(reduced, coeff)
    assert exponent == 0.0


def test_adaptive_reduction_damps_negative_sharp_result():
    coeff = np.array([0.0, -0.1, 1.0 / (2.0 * np.pi), -0.1, 0.0])
    sharp = centered_coefficients(coeff, (3,))
    reduced, exponent = adaptive_kernel_reduce_coefficients(
        coeff, (3,), kernel="korovkin", return_exponent=True
    )

    assert minimum_on_fft_grid(sharp) < 0.0
    assert exponent > 0.0
    assert minimum_on_fft_grid(reduced) >= -1e-12


@pytest.mark.parametrize("invalid_value", [1.5, np.float64(2.0), "2", True])
def test_adaptive_reduction_rejects_noninteger_search_steps(invalid_value):
    with pytest.raises(
        ValueError, match="exponent_search_steps must be a nonnegative integer"
    ):
        adaptive_kernel_reduce_coefficients(
            np.ones(3),
            exponent_search_steps=invalid_value,
        )


def test_adaptive_reduction_accepts_numpy_integer_search_steps():
    reduced = adaptive_kernel_reduce_coefficients(
        np.ones(3),
        exponent_search_steps=np.int64(0),
    )

    npt.assert_array_equal(reduced, np.ones(3))


def test_even_shape_rejected():
    with pytest.raises(ValueError):
        fejer_weights((4,))
