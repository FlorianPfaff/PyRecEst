"""Low-rank hypertoroidal Fourier distribution."""

from __future__ import annotations

from math import prod, sqrt
from numbers import Integral

import numpy as np

from ._input_validation import as_shift_vector
from ._tensor_train import TensorTrain
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from .hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution

_SHAPE_ERROR = "Fourier coefficient side lengths must be positive odd integers."


def _as_positive_odd_coefficient_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(_SHAPE_ERROR)
    value = int(value)
    if value < 1 or value % 2 != 1:
        raise ValueError(_SHAPE_ERROR)
    return value


def _as_shape(shape, *, dim=None):
    if isinstance(shape, (bool, np.bool_)):
        raise ValueError(_SHAPE_ERROR)
    if isinstance(shape, Integral):
        value = _as_positive_odd_coefficient_count(shape)
        normalized = (value,) if dim is None else (value,) * dim
    else:
        try:
            normalized = tuple(_as_positive_odd_coefficient_count(n) for n in shape)
        except TypeError as exc:
            raise TypeError(
                "shape must be an integer or a sequence of integers."
            ) from exc
    if not normalized:
        raise ValueError("shape must contain at least one dimension.")
    if dim is not None and len(normalized) != dim:
        raise ValueError(f"shape must contain {dim} dimensions.")
    return normalized


def _center(shape):
    return tuple(n // 2 for n in shape)


def _freqs(n):
    return np.arange(n, dtype=float) - n // 2


class LowRankHypertoroidalFourierDistribution(AbstractHypertoroidalDistribution):
    """Hypertoroidal Fourier distribution with TT-compressed coefficients.

    This first prototype is intended for the identity Fourier representation.
    Square-root normalization is available as an algebraic helper, but SqFF
    prediction is intentionally not implemented yet.
    """

    def __init__(self, coefficients, transformation="identity", *, normalize=True):
        if not isinstance(coefficients, TensorTrain):
            coefficients = TensorTrain(coefficients)
        _as_shape(coefficients.shape)
        if transformation not in ("identity", "sqrt"):
            raise ValueError("transformation must be 'identity' or 'sqrt'.")
        AbstractHypertoroidalDistribution.__init__(self, coefficients.ndim)
        self.coefficients = coefficients
        self.transformation = transformation
        if normalize:
            self.normalize_in_place()

    @property
    def coeff_shape(self):
        return self.coefficients.shape

    @property
    def tt_ranks(self):
        return self.coefficients.ranks

    @property
    def center_index(self):
        return _center(self.coeff_shape)

    @classmethod
    def uniform(cls, shape, transformation="identity"):
        shape = _as_shape(shape)
        coefficient = (2.0 * np.pi) ** (-len(shape))
        if transformation == "sqrt":
            coefficient = sqrt(coefficient)
        elif transformation != "identity":
            raise ValueError("transformation must be 'identity' or 'sqrt'.")
        cores = []
        for axis, axis_size in enumerate(shape):
            core = np.zeros((1, axis_size, 1), dtype=np.complex128)
            core[0, axis_size // 2, 0] = coefficient if axis == 0 else 1.0
            cores.append(core)
        return cls(TensorTrain(cores), transformation, normalize=False)

    @classmethod
    def from_dense(cls, dense, *, max_rank=None, rtol=0.0, atol=0.0):
        if isinstance(dense, HypertoroidalFourierDistribution):
            coeff = np.asarray(dense.coeff_mat, dtype=np.complex128)
            transformation = dense.transformation
        else:
            coeff = np.asarray(dense, dtype=np.complex128)
            transformation = "identity"
        tt = TensorTrain.from_dense(coeff, max_rank=max_rank, rtol=rtol, atol=atol)
        return cls(tt, transformation, normalize=True)

    @classmethod
    def from_distribution(
        cls,
        distribution,
        n_coefficients,
        desired_transformation="identity",
        *,
        max_rank=None,
        rtol=0.0,
        atol=0.0,
    ):
        dense = HypertoroidalFourierDistribution.from_distribution(
            distribution, n_coefficients, desired_transformation
        )
        return cls.from_dense(dense, max_rank=max_rank, rtol=rtol, atol=atol)

    def to_dense(self):
        return self.coefficients.to_dense()

    def coefficient_at_zero(self):
        return self.coefficients.entry(self.center_index)

    def normalize_in_place(self):
        if self.transformation == "identity":
            center_coefficient = self.coefficient_at_zero()
            if abs(center_coefficient.imag) > 0.001:
                raise ValueError("Center coefficient must be real-valued.")
            normalizer = ((2.0 * np.pi) ** self.dim) * center_coefficient.real
        else:
            normalizer = sqrt((2.0 * np.pi) ** self.dim) * self.coefficients.norm()
        if abs(normalizer) == 0:
            raise ZeroDivisionError("Cannot normalize zero coefficient tensor.")
        self.coefficients = self.coefficients.scaled(1.0 / normalizer)
        return self

    def value(self, xs):
        points = np.asarray(xs, dtype=float)
        single = False
        if self.dim == 1:
            if points.ndim == 0:
                points = points.reshape(1, 1)
                single = True
            elif points.ndim == 1:
                points = points.reshape(-1, 1)
            elif points.ndim == 2:
                if points.shape[-1] != 1:
                    raise ValueError("Expected one-dimensional points.")
            else:
                raise ValueError(
                    "Expected one-dimensional points as a scalar, vector, or column array."
                )
        else:
            if points.ndim == 0:
                raise ValueError(
                    f"Expected points with shape ({self.dim},) or (n, {self.dim})."
                )
            if points.ndim == 1:
                points = points.reshape(1, -1)
                single = True
            elif points.ndim != 2:
                raise ValueError(
                    f"Expected points with shape ({self.dim},) or (n, {self.dim})."
                )
            if points.shape[-1] != self.dim:
                raise ValueError(f"Expected points with {self.dim} columns.")

        values = np.empty(points.shape[0], dtype=np.complex128)
        frequencies = [_freqs(axis_size) for axis_size in self.coeff_shape]
        for point_index, point in enumerate(points):
            accumulated = np.ones((1, 1), dtype=np.complex128)
            for axis, (core, ks) in enumerate(
                zip(self.coefficients.cores, frequencies)
            ):
                weights = np.exp(1j * ks * point[axis])
                matrix = np.tensordot(weights, core, axes=([0], [1]))
                accumulated = accumulated @ matrix
            values[point_index] = accumulated.reshape(())
        return values[0] if single else values

    def pdf(self, xs):
        values = self.value(xs)
        if self.transformation == "identity":
            imag_abs = np.abs(np.imag(values))
            if np.any(imag_abs > 1e-10):
                raise ValueError(
                    "Density evaluation has a non-negligible imaginary part. "
                    "Check that the low-rank coefficients define a real-valued density."
                )
            return np.real(values)
        return np.real(values * np.conjugate(values))

    def shift(self, shift_by):
        shift_by = as_shift_vector(shift_by, self.dim)
        factors = [
            np.exp(-1j * _freqs(axis_size) * float(shift_by[axis]))
            for axis, axis_size in enumerate(self.coeff_shape)
        ]
        return LowRankHypertoroidalFourierDistribution(
            self.coefficients.multiply_axis_factors(factors),
            self.transformation,
            normalize=False,
        )

    def centered_hermitian_deviation(self, *, max_entries=1_000_000):
        """Return max ``|C[k] - conj(C[-k])|`` for identity coefficients."""

        if self.transformation != "identity":
            raise NotImplementedError(
                "Hermitian coefficient checks apply to identity coefficients only."
            )
        return self.coefficients.centered_hermitian_deviation(max_entries=max_entries)

    def is_centered_hermitian(self, *, atol=1e-10, max_entries=1_000_000):
        """Return whether identity coefficients define a real-valued Fourier series."""

        if self.transformation != "identity":
            raise NotImplementedError(
                "Hermitian coefficient checks apply to identity coefficients only."
            )
        return self.coefficients.is_centered_hermitian(
            atol=atol, max_entries=max_entries
        )

    def centered_hermitianized(
        self,
        *,
        max_rank=None,
        rtol=0.0,
        atol=0.0,
        max_entries=1_000_000,
    ):
        """Return a centered-Hermitian identity-coefficient distribution."""

        if self.transformation != "identity":
            raise NotImplementedError(
                "Hermitian coefficient repair applies to identity coefficients only."
            )
        coefficients = self.coefficients.centered_hermitianized(
            max_rank=max_rank,
            rtol=rtol,
            atol=atol,
            max_entries=max_entries,
        )
        return LowRankHypertoroidalFourierDistribution(
            coefficients,
            "identity",
            normalize=True,
        )

    def multiply(
        self, other, n_coefficients=None, *, max_rank=None, rtol=0.0, atol=0.0
    ):
        other = self._ensure_low_rank(other)
        self._check_compatible(other)
        target_shape = (
            self.coeff_shape
            if n_coefficients is None
            else _as_shape(n_coefficients, dim=self.dim)
        )
        coeffs = self.coefficients.coefficient_convolution(
            other.coefficients, target_shape=target_shape
        )
        coeffs = coeffs.round(max_rank=max_rank, rtol=rtol, atol=atol)
        return LowRankHypertoroidalFourierDistribution(coeffs, self.transformation)

    def convolve(
        self, other, n_coefficients=None, *, max_rank=None, rtol=0.0, atol=0.0
    ):
        other = self._ensure_low_rank(other)
        self._check_compatible(other)
        if self.transformation != "identity":
            raise NotImplementedError(
                "Low-rank SqFF prediction is not implemented yet."
            )
        if n_coefficients is not None:
            target_shape = _as_shape(n_coefficients, dim=self.dim)
            if target_shape != self.coeff_shape:
                raise NotImplementedError(
                    "Changing coefficient shape during low-rank prediction is not implemented."
                )
        coeffs = self.coefficients.hadamard_product(other.coefficients)
        coeffs = coeffs.scaled((2.0 * np.pi) ** self.dim)
        coeffs = coeffs.round(max_rank=max_rank, rtol=rtol, atol=atol)
        return LowRankHypertoroidalFourierDistribution(coeffs, "identity")

    def mean_direction(self):
        if self.transformation != "identity":
            raise NotImplementedError(
                "mean_direction is implemented for identity coefficients only."
            )
        scale = (2.0 * np.pi) ** self.dim
        means = np.zeros(self.dim)
        for axis in range(self.dim):
            index = list(self.center_index)
            index[axis] -= 1
            moment = scale * self.coefficients.entry(index)
            means[axis] = np.mod(np.angle(moment), 2.0 * np.pi)
        return means

    def integrate(self):
        if self.transformation == "identity":
            return float(
                np.real_if_close(
                    ((2.0 * np.pi) ** self.dim) * self.coefficient_at_zero()
                ).real
            )
        return float(((2.0 * np.pi) ** self.dim) * self.coefficients.norm_squared())

    def compression_ratio(self):
        return prod(self.coeff_shape) / self.coefficients.storage_size

    def _ensure_low_rank(self, other):
        if isinstance(other, LowRankHypertoroidalFourierDistribution):
            return other
        if isinstance(other, HypertoroidalFourierDistribution):
            return LowRankHypertoroidalFourierDistribution.from_dense(other)
        raise TypeError(
            "Expected a dense or low-rank hypertoroidal Fourier distribution."
        )

    def _check_compatible(self, other):
        if self.coeff_shape != other.coeff_shape:
            raise ValueError(
                "Fourier distributions must have identical coefficient shapes."
            )
        if self.transformation != other.transformation:
            raise ValueError("Fourier distributions must use the same transformation.")
