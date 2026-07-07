"""Private tensor-train helper for hypertoroidal Fourier coefficients."""

from __future__ import annotations

from math import prod, sqrt

import numpy as np
from scipy import signal


def _choose_rank(singular_values, max_rank, local_tolerance):
    full_rank = singular_values.size
    if local_tolerance <= 0:
        rank = full_rank
    else:
        squared_tail = np.cumsum(singular_values[::-1] ** 2)[::-1]
        rank = full_rank
        for candidate in range(1, full_rank + 1):
            tail = 0.0 if candidate == full_rank else sqrt(float(squared_tail[candidate]))
            if tail <= local_tolerance:
                rank = candidate
                break
    if max_rank is not None:
        if max_rank < 1:
            raise ValueError("max_rank must be positive when provided.")
        rank = min(rank, max_rank)
    return max(1, rank)


class TensorTrain:
    """Minimal complex tensor-train representation."""

    def __init__(self, cores):
        checked = tuple(np.asarray(core, dtype=np.complex128).copy() for core in cores)
        if not checked:
            raise ValueError("At least one TT core is required.")
        for core in checked:
            if core.ndim != 3:
                raise ValueError("TT cores must have shape (left_rank, mode_size, right_rank).")
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
        array = np.asarray(tensor, dtype=np.complex128)
        if array.ndim < 1:
            raise ValueError("A tensor with at least one axis is required.")
        if any(axis_size < 1 for axis_size in array.shape):
            raise ValueError("All tensor axes must be non-empty.")
        if rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be non-negative.")
        if array.ndim == 1:
            return cls((array.reshape(1, array.shape[0], 1),))

        norm = float(np.linalg.norm(array.ravel()))
        global_tolerance = max(float(atol), float(rtol) * norm)
        local_tolerance = global_tolerance / sqrt(array.ndim - 1) if global_tolerance > 0 else 0.0

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
        value = self.cores[0][:, int(multi_index[0]), :]
        for axis, index in enumerate(multi_index[1:], start=1):
            value = value @ self.cores[axis][:, int(index), :]
        return complex(value.reshape(()))

    def norm_squared(self):
        dense = self.to_dense().ravel()
        value = np.vdot(dense, dense)
        return float(np.real_if_close(value, tol=1000).real)

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
                raise ValueError("Each factor vector must match the corresponding mode size.")
            cores.append(core * vector[None, :, None])
        return TensorTrain(cores)

    def hadamard_product(self, other):
        if self.shape != other.shape:
            raise ValueError("Hadamard products require identical tensor shapes.")
        return TensorTrain.from_dense(self.to_dense() * other.to_dense())

    def coefficient_convolution(self, other, *, target_shape=None):
        if self.ndim != other.ndim:
            raise ValueError("Convolution operands must have the same number of dimensions.")
        if target_shape is None:
            target_shape = self.shape
        conv = signal.fftconvolve(self.to_dense(), other.to_dense(), mode="same")
        if tuple(conv.shape) != tuple(target_shape):
            slices = tuple(slice(0, int(n)) for n in target_shape)
            conv = conv[slices]
        return TensorTrain.from_dense(conv)

    def round(self, *, max_rank=None, rtol=0.0, atol=0.0, max_dense_entries=1_000_000):
        if self.size > max_dense_entries and (max_rank is not None or rtol > 0 or atol > 0):
            raise ValueError("Dense TT-SVD fallback would exceed max_dense_entries.")
        if self.size > max_dense_entries:
            return self.copy()
        return TensorTrain.from_dense(self.to_dense(), max_rank=max_rank, rtol=rtol, atol=atol)
