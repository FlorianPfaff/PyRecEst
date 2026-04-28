"""Chordal-mean smoother for SO(3) rotation sequences."""

from __future__ import annotations

from typing import Sequence

# pylint: disable=no-member
from pyrecest import backend
from pyrecest.backend import asarray, diag, linalg, ndim, zeros

from .abstract_smoother import AbstractSmoother


class SO3ChordalMeanSmoother(AbstractSmoother):
    """Smooth SO(3) rotation sequences with local weighted chordal means.

    Rotations are represented as 3-by-3 rotation matrices. The chordal mean is
    computed by averaging rotation matrices in the ambient Euclidean space and
    projecting the result back onto SO(3) with the orthogonal Procrustes
    solution.

    Parameters
    ----------
    window_size
        Number of neighboring rotations used per local mean. The window is
        centered as far as possible around each time index and clipped at the
        sequence boundaries.
    kernel_weights
        Optional nonnegative weights for positions inside the local window. If
        provided, its length must match ``window_size``.
    """

    def __init__(self, window_size: int = 3, kernel_weights=None):
        self.window_size = self._validate_window_size(window_size)
        self.kernel_weights = self._normalize_weight_vector(
            kernel_weights,
            self.window_size,
            "kernel_weights",
            normalize=False,
        )

    @staticmethod
    def _validate_window_size(window_size: int) -> int:
        try:
            window_size_int = int(window_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("window_size must be a positive integer.") from exc

        if window_size_int < 1:
            raise ValueError("window_size must be a positive integer.")
        return window_size_int

    @staticmethod
    def _as_rotation_list(rotations) -> list:
        if isinstance(rotations, (list, tuple)) and len(rotations) == 0:
            return []

        if isinstance(rotations, (list, tuple)):
            rotation_list = [asarray(rotation) for rotation in rotations]
            if all(
                ndim(rotation) == 2 and rotation.shape == (3, 3)
                for rotation in rotation_list
            ):
                return rotation_list

        rotation_array = asarray(rotations)

        if ndim(rotation_array) == 2:
            if rotation_array.shape != (3, 3):
                raise ValueError("A single SO(3) rotation must have shape (3, 3).")
            return [rotation_array]

        if ndim(rotation_array) == 3:
            if rotation_array.shape[1:] == (3, 3):
                return [rotation_array[idx] for idx in range(rotation_array.shape[0])]
            if rotation_array.shape[:2] == (3, 3):
                return [
                    rotation_array[:, :, idx] for idx in range(rotation_array.shape[2])
                ]

        raise ValueError(
            "rotations must be a rotation matrix, a sequence of rotation matrices, "
            "or an array with shape (n, 3, 3) or (3, 3, n)."
        )

    @staticmethod
    def _normalize_weight_vector(
        weights,
        length: int,
        name: str,
        normalize: bool = True,
    ):
        if weights is None:
            return None

        weights_array = asarray(weights).reshape(-1)
        if weights_array.shape[0] != length:
            raise ValueError(f"{name} must have length {length}.")

        for idx in range(length):
            if weights_array[idx] < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

        weight_sum = backend.sum(weights_array)
        if weight_sum <= 0.0:
            raise ValueError(f"{name} must contain at least one positive entry.")

        if normalize:
            return weights_array / weight_sum
        return weights_array

    @staticmethod
    def project_to_so3(matrix):
        """Project a 3-by-3 matrix to the closest SO(3) rotation matrix."""
        matrix = asarray(matrix)
        if matrix.shape != (3, 3):
            raise ValueError("matrix must have shape (3, 3).")

        left_singular_vectors, _, right_singular_vectors_transposed = linalg.svd(matrix)
        determinant = linalg.det(
            left_singular_vectors @ right_singular_vectors_transposed
        )
        correction = diag(asarray([1.0, 1.0, 1.0 if determinant >= 0.0 else -1.0]))
        return left_singular_vectors @ correction @ right_singular_vectors_transposed

    @staticmethod
    def chordal_distance(rotation_a, rotation_b):
        """Return the Frobenius chordal distance between two SO(3) rotations."""
        rotation_a = asarray(rotation_a)
        rotation_b = asarray(rotation_b)
        if rotation_a.shape != (3, 3) or rotation_b.shape != (3, 3):
            raise ValueError("Both rotations must have shape (3, 3).")
        return linalg.norm(rotation_a - rotation_b)

    @classmethod
    def chordal_mean(cls, rotations, weights=None):
        """Compute the weighted chordal mean of one or more SO(3) rotations."""
        rotation_list = cls._as_rotation_list(rotations)
        if len(rotation_list) == 0:
            raise ValueError("At least one rotation is required.")

        normalized_weights = cls._normalize_weight_vector(
            weights,
            len(rotation_list),
            "weights",
        )
        if normalized_weights is None:
            normalized_weights = asarray(
                [1.0 / len(rotation_list) for _ in rotation_list]
            )

        mean_matrix = zeros((3, 3))
        for idx, rotation in enumerate(rotation_list):
            mean_matrix = mean_matrix + normalized_weights[idx] * rotation

        return cls.project_to_so3(mean_matrix)

    def _active_parameters(self, window_size):
        if window_size is None:
            return self.window_size, self.kernel_weights
        return self._validate_window_size(window_size), None

    @staticmethod
    def _window_bounds(sequence_length: int, window_size: int, idx: int):
        before = window_size // 2
        after = window_size - before - 1
        start = max(0, idx - before)
        stop = min(sequence_length, idx + after + 1)
        first_kernel_index = before - (idx - start)
        return start, stop, first_kernel_index

    def _local_weights(
        self,
        window_bounds,
        sample_weights,
        kernel_weights,
    ):
        if sample_weights is None and kernel_weights is None:
            return None

        start, stop, first_kernel_index = window_bounds
        local_weights = []
        for rotation_idx in range(start, stop):
            weight = 1.0
            if sample_weights is not None:
                weight = sample_weights[rotation_idx]
            if kernel_weights is not None:
                kernel_idx = first_kernel_index + rotation_idx - start
                weight = weight * kernel_weights[kernel_idx]
            local_weights.append(weight)
        return asarray(local_weights)

    def smooth(
        self,
        rotations: Sequence,
        weights=None,
        window_size: int | None = None,
    ) -> list:
        """Smooth a rotation sequence with local chordal means.

        Parameters
        ----------
        rotations
            Rotation matrix sequence as a Python sequence, ``(n, 3, 3)`` array,
            or ``(3, 3, n)`` array.
        weights
            Optional nonnegative reliability weights, one per input rotation.
        window_size
            Optional per-call override for the number of rotations in each local
            mean. Kernel weights from construction are only used when this is not
            overridden.

        Returns
        -------
        list
            Smoothed SO(3) rotations, one per input rotation.
        """
        rotation_list = self._as_rotation_list(rotations)
        if len(rotation_list) == 0:
            return []

        active_window_size, active_kernel_weights = self._active_parameters(window_size)
        sample_weights = self._normalize_weight_vector(
            weights,
            len(rotation_list),
            "weights",
            normalize=False,
        )

        smoothed = []
        for idx in range(len(rotation_list)):
            window_bounds = self._window_bounds(
                len(rotation_list), active_window_size, idx
            )
            local_weights = self._local_weights(
                window_bounds,
                sample_weights,
                active_kernel_weights,
            )
            start, stop, _ = window_bounds
            smoothed.append(self.chordal_mean(rotation_list[start:stop], local_weights))

        return smoothed


SO3CMSmoother = SO3ChordalMeanSmoother
