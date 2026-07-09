"""Utilities for centered Fourier coefficient tensors and positive-kernel reductions."""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Literal

import numpy as np

CoefficientShape = int | Iterable[int]
ReductionKernel = Literal["sharp", "none", "fejer", "korovkin", "fejer-korovkin", "fk"]


def normalize_coefficient_shape(
    shape_like: CoefficientShape,
    *,
    dim: int | None = None,
    name: str = "n_coefficients",
) -> tuple[int, ...]:
    """Validate and normalize a centered Fourier coefficient shape.

    PyRecEst's hypertoroidal Fourier coefficients use odd side lengths so that
    the central tensor entry is the zero-frequency coefficient. A scalar shape
    is interpreted as a one-dimensional shape unless ``dim`` is given; with
    ``dim`` it is broadcast to all dimensions.
    """

    if isinstance(shape_like, bool):
        raise ValueError(f"{name} must contain positive odd integers.")

    if isinstance(shape_like, Integral):
        values = (int(shape_like),) if dim is None else (int(shape_like),) * dim
    else:
        try:
            values = tuple(int(value) for value in shape_like)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an integer or an iterable of integers."
            ) from exc

    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one entry.")
    if dim is not None and len(values) != dim:
        raise ValueError(f"{name} must contain {dim} entries.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} entries must be positive.")
    if any(value % 2 != 1 for value in values):
        raise ValueError(f"{name} entries must be odd in every dimension.")

    return values


def normalize_kernel_name(kernel: str) -> str:
    """Normalize aliases for supported coefficient-reduction kernels."""

    normalized = kernel.lower().replace("_", "-")
    if normalized in ("none", "sharp"):
        return "sharp"
    if normalized in ("fejer", "cesaro"):
        return "fejer"
    if normalized in ("korovkin", "fejer-korovkin", "fk"):
        return "korovkin"
    raise ValueError(
        f"Unsupported reduction kernel {kernel!r}. Use 'sharp', 'fejer', or 'korovkin'."
    )


def centered_coefficients(coefficients, target_shape: CoefficientShape):
    """Center-crop or center-pad a Fourier coefficient tensor.

    Parameters
    ----------
    coefficients : array-like
        Centered coefficient tensor. Every axis must have odd length.
    target_shape : int or iterable of int
        Desired centered tensor shape. Every axis must have odd length.
    """

    coeff_arr = np.asarray(coefficients)
    target_shape = normalize_coefficient_shape(
        target_shape, dim=coeff_arr.ndim, name="target_shape"
    )
    current_shape = normalize_coefficient_shape(
        coeff_arr.shape, dim=coeff_arr.ndim, name="coefficients.shape"
    )

    if current_shape == target_shape:
        return coeff_arr.copy()

    result = np.zeros(target_shape, dtype=coeff_arr.dtype)
    old_slices = []
    new_slices = []
    for old_len, new_len in zip(current_shape, target_shape):
        overlap = min(old_len, new_len)
        old_start = (old_len - overlap) // 2
        new_start = (new_len - overlap) // 2
        old_slices.append(slice(old_start, old_start + overlap))
        new_slices.append(slice(new_start, new_start + overlap))

    result[tuple(new_slices)] = coeff_arr[tuple(old_slices)]
    return result


def _product_weights(
    shape: tuple[int, ...], one_dimensional_weight_factory, *, dtype=float
):
    weights = np.ones(shape, dtype=dtype)
    for axis, side_length in enumerate(shape):
        one_dim = one_dimensional_weight_factory(side_length, dtype=dtype)
        reshape_shape = [1] * len(shape)
        reshape_shape[axis] = side_length
        weights = weights * one_dim.reshape(reshape_shape)
    return weights


def _fejer_weights_1d(side_length: int, *, dtype=float):
    order = (side_length - 1) // 2
    if order == 0:
        return np.ones((1,), dtype=dtype)
    ks = np.arange(-order, order + 1, dtype=dtype)
    return 1.0 - np.abs(ks) / (order + 1.0)


def _korovkin_weights_1d(side_length: int, *, dtype=float):
    """Return the one-dimensional Fejer-Korovkin multiplier sequence.

    For order ``K`` and ``a = pi/(K + 2)``, the multiplier for ``|k| <= K`` is

    ``((K + 1 - |k|) cos(|k| a) + sin((|k| + 1) a) / sin(a)) / (K + 2)``.

    This is the autocorrelation sequence of the normalized sine vector
    ``sqrt(2/(K+2)) * sin(j*a)``, ``j = 1, ..., K+1``. Hence the associated
    trigonometric kernel is nonnegative. Its first multiplier is exactly
    ``cos(pi/(K+2))``.
    """

    order = (side_length - 1) // 2
    if order == 0:
        return np.ones((1,), dtype=dtype)

    abs_ks = np.abs(np.arange(-order, order + 1, dtype=dtype))
    angle = np.pi / (order + 2.0)
    weights = (
        (order + 1.0 - abs_ks) * np.cos(abs_ks * angle)
        + np.sin((abs_ks + 1.0) * angle) / np.sin(angle)
    ) / (order + 2.0)

    # Remove harmless floating-point drift at the zero-frequency coefficient.
    weights[order] = 1.0
    return weights.astype(dtype, copy=False)


def fejer_weights(shape: CoefficientShape, *, dtype=float):
    """Return separable Fejer/Cesaro weights for centered Fourier coefficients.

    For each side length ``n = 2*K + 1`` the one-dimensional weights are
    ``1 - abs(k)/(K + 1)`` for ``k = -K, ..., K``. For multidimensional
    tensors the returned weights are the tensor product of the one-dimensional
    weights. The central weight is exactly one, so the zero-frequency
    coefficient and therefore the integral are unchanged.
    """

    shape = normalize_coefficient_shape(shape, name="shape")
    return _product_weights(shape, _fejer_weights_1d, dtype=dtype)


def korovkin_weights(shape: CoefficientShape, *, dtype=float):
    """Return separable Fejer-Korovkin weights for centered coefficients.

    The tensor-product kernel is nonnegative because each one-dimensional kernel
    is generated by an autocorrelation sequence and products of nonnegative
    kernels remain nonnegative. The first one-dimensional multiplier is
    ``cos(pi/(K+2))`` for order ``K``, so low-frequency bias is second-order in
    ``K`` instead of first-order as for the plain Fejer weights.
    """

    shape = normalize_coefficient_shape(shape, name="shape")
    return _product_weights(shape, _korovkin_weights_1d, dtype=dtype)


def positive_kernel_weights(
    shape: CoefficientShape, *, kernel: str = "fejer", dtype=float
):
    """Return tensor-product weights for a supported coefficient-reduction kernel."""

    kernel = normalize_kernel_name(kernel)
    shape = normalize_coefficient_shape(shape, name="shape")
    if kernel == "sharp":
        return np.ones(shape, dtype=dtype)
    if kernel == "fejer":
        return fejer_weights(shape, dtype=dtype)
    if kernel == "korovkin":
        return korovkin_weights(shape, dtype=dtype)
    raise AssertionError(f"Unhandled normalized kernel {kernel!r}.")


def apply_kernel_weights(coefficients, *, kernel: str = "fejer", exponent: float = 1.0):
    """Apply separable positive-kernel weights to a centered coefficient tensor.

    ``exponent`` is intended for adaptive damping. An exponent of zero gives
    sharp truncation, while an exponent of one gives the full selected kernel.
    Intermediate exponents are a practical grid-level safeguard and should not
    be interpreted as an analytic nonnegativity certificate.
    """

    if exponent < 0.0:
        raise ValueError("exponent must be nonnegative.")
    coeff_arr = np.asarray(coefficients)
    normalize_coefficient_shape(
        coeff_arr.shape, dim=coeff_arr.ndim, name="coefficients.shape"
    )
    kernel = normalize_kernel_name(kernel)
    if kernel == "sharp" or exponent == 0.0:
        return coeff_arr.copy()
    weights = positive_kernel_weights(coeff_arr.shape, kernel=kernel, dtype=float)
    if exponent != 1.0:
        weights = np.power(weights, exponent)
    return coeff_arr * weights


def apply_fejer_weights(coefficients):
    """Apply separable Fejer weights to a centered coefficient tensor."""

    return apply_kernel_weights(coefficients, kernel="fejer")


def reduce_coefficients(
    coefficients,
    target_shape: CoefficientShape | None = None,
    *,
    kernel: str = "fejer",
    exponent: float = 1.0,
):
    """Reduce centered coefficients by center alignment followed by kernel weights."""

    coeff_arr = np.asarray(coefficients)
    if target_shape is None:
        target_shape = coeff_arr.shape
    reduced = centered_coefficients(coeff_arr, target_shape)
    return apply_kernel_weights(reduced, kernel=kernel, exponent=exponent)


def fejer_reduce_coefficients(
    coefficients, target_shape: CoefficientShape | None = None
):
    """Reduce centered coefficients by center alignment followed by Fejer weights."""

    return reduce_coefficients(coefficients, target_shape, kernel="fejer")


def coefficient_grid_shape(
    shape: CoefficientShape, oversampling_factor: int = 1
) -> tuple[int, ...]:
    """Return an odd FFT-grid shape obtained by centered zero-padding."""

    if isinstance(oversampling_factor, bool) or int(oversampling_factor) < 1:
        raise ValueError("oversampling_factor must be a positive integer.")
    oversampling_factor = int(oversampling_factor)
    shape = normalize_coefficient_shape(shape, name="shape")
    return tuple((side_length - 1) * oversampling_factor + 1 for side_length in shape)


def values_on_fft_grid(coefficients, grid_shape: CoefficientShape | None = None):
    """Evaluate centered Fourier coefficients on their equidistant FFT grid."""

    coeff_arr = np.asarray(coefficients)
    normalize_coefficient_shape(
        coeff_arr.shape, dim=coeff_arr.ndim, name="coefficients.shape"
    )
    if grid_shape is not None:
        grid_shape = normalize_coefficient_shape(
            grid_shape, dim=coeff_arr.ndim, name="grid_shape"
        )
        coeff_arr = centered_coefficients(coeff_arr, grid_shape)
    values = np.fft.ifftn(np.fft.ifftshift(coeff_arr)) * np.prod(coeff_arr.shape)
    return np.real_if_close(values, tol=1000).real


def minimum_on_fft_grid(
    coefficients, grid_shape: CoefficientShape | None = None
) -> float:
    """Return the minimum real value on an equidistant FFT diagnostic grid."""

    return float(np.min(values_on_fft_grid(coefficients, grid_shape=grid_shape)))


def adaptive_kernel_reduce_coefficients(
    coefficients,
    target_shape: CoefficientShape | None = None,
    *,
    kernel: str = "korovkin",
    min_value_tolerance: float = 1e-12,
    oversampling_factor: int = 1,
    exponent_search_steps: int = 24,
    return_exponent: bool = False,
):
    """Reduce coefficients adaptively, damping only if grid negativity appears.

    The routine first performs sharp center reduction. If the resulting
    trigonometric polynomial is nonnegative on the diagnostic FFT grid up to
    ``min_value_tolerance``, the sharp coefficients are returned unchanged. If
    grid negativity appears, the selected positive-kernel weights are applied.
    When the full kernel clears the grid-level negativity, a bisection search is
    used to find a smaller exponent ``theta`` in ``weights**theta``.

    Only the full positive-kernel case is an analytic positivity-preserving
    convolution reduction. Intermediate exponents are a practical safeguard that
    is certified only on the diagnostic grid.
    """

    if min_value_tolerance < 0.0:
        raise ValueError("min_value_tolerance must be nonnegative.")
    if isinstance(exponent_search_steps, bool) or int(exponent_search_steps) < 0:
        raise ValueError("exponent_search_steps must be a nonnegative integer.")

    coeff_arr = np.asarray(coefficients)
    if target_shape is None:
        target_shape = coeff_arr.shape
    target_shape = normalize_coefficient_shape(
        target_shape, dim=coeff_arr.ndim, name="target_shape"
    )
    diagnostic_shape = coefficient_grid_shape(
        target_shape, oversampling_factor=oversampling_factor
    )

    sharp = centered_coefficients(coeff_arr, target_shape)
    if minimum_on_fft_grid(sharp, diagnostic_shape) >= -min_value_tolerance:
        return (sharp, 0.0) if return_exponent else sharp

    full = reduce_coefficients(coeff_arr, target_shape, kernel=kernel, exponent=1.0)
    if (
        minimum_on_fft_grid(full, diagnostic_shape) < -min_value_tolerance
        or exponent_search_steps == 0
    ):
        return (full, 1.0) if return_exponent else full

    low = 0.0
    high = 1.0
    best = full
    for _ in range(int(exponent_search_steps)):
        mid = 0.5 * (low + high)
        candidate = reduce_coefficients(
            coeff_arr, target_shape, kernel=kernel, exponent=mid
        )
        if minimum_on_fft_grid(candidate, diagnostic_shape) >= -min_value_tolerance:
            best = candidate
            high = mid
        else:
            low = mid

    return (best, high) if return_exponent else best
