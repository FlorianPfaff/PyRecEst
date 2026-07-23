"""Private tensor-train helper for hypertoroidal Fourier coefficients."""

from __future__ import annotations

from math import prod, sqrt
from operator import index as _operator_index

import numpy as np


def _normalize_max_rank(max_rank):
    if max_rank is None:
        return None
    if isinstance(max_rank, (bool, np.bool_)):
        raise TypeError("max_rank must be an integer when provided.")
    try:
        normalized = _operator_index(max_rank)
    except TypeError as exc:
        raise TypeError("max_rank must be an integer when provided.") from exc
    if normalized < 1:
        raise ValueError("max_rank must be positive when provided.")
    return normalized


def _normalize_nonnegative_tolerance(value, name):
    """Return a finite non-negative scalar tolerance."""
    message = f"{name} must be a non-negative finite real scalar."
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(message)
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(message) from exc
    if values.shape != () or values.dtype.kind not in "iuf":
        raise TypeError(message)
    tolerance = float(values)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(message)
    return tolerance


def _check_dense_validation_size(size, max_entries):
    if max_entries is None:
        return
    max_entries = _operator_index(max_entries)
    if max_entries < 1:
        raise ValueError("max_entries must be positive when provided.")
    if size > max_entries:
        raise ValueError(
            "Centered Hermitian validation would require visiting more than "
            f"{max_entries} tensor entries. Increase max_entries explicitly."
        )


def _choose_rank(singular_values, max_rank, local_tolerance):
    max_rank = _normalize_max_rank(max_rank)
    full_rank = singular_values.size
    if local_tolerance <= 0:
        rank = full_rank
    else:
        squared_tail = np.cumsum(singular_values[::-1] ** 2)[::-1]
        rank = full_rank
        for candidate in range(1, full_rank + 1):
            tail = (
                0.0 if candidate == full_rank else sqrt(float(squared_tail[candidate]))
            )
            if tail <= local_tolerance:
                rank = candidate
                break
    if max_rank is not None:
        rank = min(rank, max_rank)
    return max(1, rank)


def _as_integer_index(value, axis, mode_size):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("TT entry indices must be integers.")
    index = _operator_index(value)
    if not -mode_size <= index < mode_size:
        raise IndexError(
            f"TT entry index {index} is out of bounds for axis {axis} with size {mode_size}."
        )
    return index


def _conjugate_flipped_centered(dense):
    axes = tuple(range(dense.ndim))
    return np.conjugate(np.flip(dense, axis=axes))


class TensorTrain:
    """Minimal complex tensor-train representation."""

    def __init__(self, cores):
        checked = tuple(np.asarray(core, dtype=np.complex128).copy() for core in cores)
        if not checked:
            raise ValueError("At least one TT core is required.")
        for core in checked:
            if core.ndim != 3:
                raise ValueError(
                    "TT cores must have shape (left_rank, mode_size, right_rank)."
                )
        if checked[0].shape[0] != 1 or checked[-1].shape[2] != 1:
            raise ValueError("Boundary TT ranks must be one.")
        for left, right in zip(checked, checked[1:]):
            if left.shape[2] != right.shape[0]:
                raise ValueError("Adjacent TT ranks do not match.")
        self.cores = checked

    @property
    def ndim(self):
        return len(self.cores)

    @property
    def shape(self):
        return tuple(core.shape[1] for core in self.cores)

    @property
    def ranks(self):
        return (self.cores[0].shape[0],) + tuple(core.shape[2] for core in self.cores)

    @property
    def size(self):
        return prod(self.shape)

    @property
    def storage_size(self):
        return int(sum(core.size for core in self.cores))

    @classmethod
    def from_dense(cls, tensor, *, max_rank=None, rtol=0.0, atol=0.0):
        max_rank = _normalize_max_rank(max_rank)
        rtol = _normalize_nonnegative_tolerance(rtol, "rtol")
        atol = _normalize_nonnegative_tolerance(atol, "atol")
        array = np.asarray(tensor, dtype=np.complex128)
        if array.ndim < 1:
            raise ValueError("A tensor with at least one axis is required.")
        if any(axis_size < 1 for axis_size in array.shape):
            raise ValueError("All tensor axes must be non-empty.")
        if array.ndim == 1:
            return cls((array.reshape(1, array.shape[0], 1),))

        norm = float(np.linalg.norm(array.ravel()))
        global_tolerance = max(atol, rtol * norm)
        local_tolerance = (
            global_tolerance / sqrt(array.ndim - 1) if global_tolerance > 0 else 0.0
        )

        cores = []
        unfolding = array
        left_rank = 1
        for mode_size in array.shape[:-1]:
            matrix = unfolding.reshape(left_rank * mode_size, -1)
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            rank = _choose_rank(singular_values, max_rank, local_tolerance)
            cores.append(u[:, :rank].reshape(left_rank, mode_size, rank))
            unfolding = singular_values[:rank, None] * vh[:rank, :]
            left_rank = rank
        cores.append(unfolding.reshape(left_rank, array.shape[-1], 1))
        return cls(cores)

    def copy(self):
        return TensorTrain(tuple(core.copy() for core in self.cores))

    def to_dense(self):
        result = self.cores[0][0, :, :]
        for core in self.cores[1:]:
            result = np.tensordot(result, core, axes=([-1], [0]))
        return np.squeeze(result, axis=-1)

    def entry(self, multi_index):
        if len(multi_index) != self.ndim:
            raise ValueError("multi_index must contain one index per TT core.")
        first_index = _as_integer_index(multi_index[0], 0, self.cores[0].shape[1])
        value = self.cores[0][:, first_index, :]
        for axis, index in enumerate(multi_index[1:], start=1):
            axis_index = _as_integer_index(index, axis, self.cores[axis].shape[1])
            value = value @ self.cores[axis][:, axis_index, :]
        return complex(value.reshape(()))

    def norm_squared(self):
        environment = np.ones((1, 1), dtype=np.complex128)
        for core in self.cores:
            next_environment = np.zeros(
                (core.shape[2], core.shape[2]), dtype=np.complex128
            )
            for mode_index in range(core.shape[1]):
                core_slice = core[:, mode_index, :]
                next_environment += core_slice.conj().T @ environment @ core_slice
            environment = next_environment
        return float(np.real_if_close(environment.reshape(()), tol=1000).real)

    def norm(self):
        return sqrt(max(self.norm_squared(), 0.0))

    def scaled(self, factor):
        cores = [core.copy() for core in self.cores]
        cores[0] = cores[0] * factor
        return TensorTrain(cores)

    def multiply_axis_factors(self, factors):
        if len(factors) != self.ndim:
            raise ValueError("factors must contain one vector per TT core.")
        cores = []
        for core, factor in zip(self.cores, factors):
            vector = np.asarray(factor, dtype=np.complex128)
            if vector.shape != (core.shape[1],):
                raise ValueError(
                    "Each factor vector must match the corresponding mode size."
                )
            cores.append(core * vector[None, :, None])
        return TensorTrain(cores)

    def hadamard_product(self, other):
        if self.shape != other.shape:
            raise ValueError("Hadamard products require identical tensor shapes.")
        cores = []
        for left_core, right_core in zip(self.cores, other.cores):
            ra0, mode_size, ra1 = left_core.shape
            rb0, _, rb1 = right_core.shape
            combined = np.zeros((ra0 * rb0, mode_size, ra1 * rb1), dtype=np.complex128)
            for mode_index in range(mode_size):
                combined[:, mode_index, :] = np.kron(
                    left_core[:, mode_index, :], right_core[:, mode_index, :]
                )
            cores.append(combined)
        return TensorTrain(cores)

    def coefficient_convolution(self, other, *, target_shape=None):
        if self.ndim != other.ndim:
            raise ValueError(
                "Convolution operands must have the same number of dimensions."
            )
        if target_shape is None:
            target_shape = self.shape
        if len(target_shape) != self.ndim:
            raise ValueError("target_shape must contain one mode size per dimension.")
        cores = [
            _convolve_cores_centered(left_core, right_core, int(mode_size))
            for left_core, right_core, mode_size in zip(
                self.cores, other.cores, target_shape
            )
        ]
        return TensorTrain(cores)

    def centered_hermitian_deviation(self, *, max_entries=1_000_000):
        """Return max ``|C[k] - conj(C[-k])|`` for center-indexed coefficients."""

        _check_dense_validation_size(self.size, max_entries)
        dense = self.to_dense()
        return float(np.max(np.abs(dense - _conjugate_flipped_centered(dense))))

    def is_centered_hermitian(self, *, atol=1e-10, max_entries=1_000_000):
        """Return whether center-indexed coefficients satisfy Hermitian symmetry."""

        atol = _normalize_nonnegative_tolerance(atol, "atol")
        return self.centered_hermitian_deviation(max_entries=max_entries) <= atol

    def centered_hermitianized(
        self,
        *,
        max_rank=None,
        rtol=0.0,
        atol=0.0,
        max_entries=1_000_000,
    ):
        """Return the nearest dense-average centered-Hermitian TT representation."""

        _check_dense_validation_size(self.size, max_entries)
        dense = self.to_dense()
        hermitian_dense = 0.5 * (dense + _conjugate_flipped_centered(dense))
        return TensorTrain.from_dense(
            hermitian_dense, max_rank=max_rank, rtol=rtol, atol=atol
        )

    def round(self, *, max_rank=None, rtol=0.0, atol=0.0, max_dense_entries=None):
        """Return a rank-rounded TT without materializing the full tensor.

        The optional ``max_dense_entries`` argument is accepted for backward
        compatibility with the first prototype but is no longer used. Rounding
        is performed by right-orthogonalization followed by left-to-right SVD
        truncation. With the default parameters, the operation preserves the
        represented tensor up to numerical roundoff while canonicalizing ranks.
        """

        del max_dense_entries
        max_rank = _normalize_max_rank(max_rank)
        rtol = _normalize_nonnegative_tolerance(rtol, "rtol")
        atol = _normalize_nonnegative_tolerance(atol, "atol")
        if self.ndim == 1:
            return self.copy()

        norm = self.norm()
        global_tolerance = max(atol, rtol * norm)
        local_tolerance = (
            global_tolerance / sqrt(self.ndim - 1) if global_tolerance > 0 else 0.0
        )

        cores = [core.copy() for core in self.cores]

        # Right-orthogonalize cores 1, ..., d-1. The R factors are absorbed
        # into the preceding core so that the left-to-right truncation sweep can
        # use ordinary matrix SVDs at each bond.
        for axis in range(self.ndim - 1, 0, -1):
            core = cores[axis]
            left_rank, mode_size, right_rank = core.shape
            matrix = core.reshape(left_rank, mode_size * right_rank)
            q_trans, r_trans = np.linalg.qr(matrix.T, mode="reduced")
            new_left_rank = q_trans.shape[1]
            cores[axis] = q_trans.T.reshape(new_left_rank, mode_size, right_rank)
            transfer = r_trans.T
            cores[axis - 1] = np.tensordot(cores[axis - 1], transfer, axes=([2], [0]))

        # Sweep left to right, truncate every bond, and absorb singular values
        # into the following core.
        for axis in range(self.ndim - 1):
            core = cores[axis]
            left_rank, mode_size, right_rank = core.shape
            matrix = core.reshape(left_rank * mode_size, right_rank)
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            rank = _choose_rank(singular_values, max_rank, local_tolerance)
            cores[axis] = u[:, :rank].reshape(left_rank, mode_size, rank)
            transfer = singular_values[:rank, None] * vh[:rank, :]
            cores[axis + 1] = np.tensordot(transfer, cores[axis + 1], axes=([1], [0]))

        return TensorTrain(cores)


def _convolve_cores_centered(left_core, right_core, target_mode_size):
    if target_mode_size < 1:
        raise ValueError("target mode sizes must be positive.")
    left_center = left_core.shape[1] // 2
    right_center = right_core.shape[1] // 2
    target_center = target_mode_size // 2
    ra0, _, ra1 = left_core.shape
    rb0, _, rb1 = right_core.shape
    result = np.zeros((ra0 * rb0, target_mode_size, ra1 * rb1), dtype=np.complex128)
    for ell_index in range(target_mode_size):
        ell_frequency = ell_index - target_center
        slice_sum = np.zeros((ra0 * rb0, ra1 * rb1), dtype=np.complex128)
        for left_index in range(left_core.shape[1]):
            left_frequency = left_index - left_center
            right_frequency = ell_frequency - left_frequency
            right_index = right_frequency + right_center
            if 0 <= right_index < right_core.shape[1]:
                slice_sum += np.kron(
                    left_core[:, left_index, :], right_core[:, int(right_index), :]
                )
        result[:, ell_index, :] = slice_sum
    return result
