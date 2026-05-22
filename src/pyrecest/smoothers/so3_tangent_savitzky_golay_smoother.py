"""Tangent-space Savitzky-Golay smoother for SO(3) rotation sequences."""

from __future__ import annotations

import math
from collections.abc import Sequence

# pylint: disable=no-member,too-many-locals
from pyrecest.backend import array, asarray, eye, linalg, ndim, reshape, stack, zeros

from .abstract_smoother import AbstractSmoother
from .so3_chordal_mean_smoother import SO3ChordalMeanSmoother


class SO3TangentSavitzkyGolaySmoother(AbstractSmoother):
    """Smooth SO(3) sequences with local tangent-space polynomial fits.

    This smoother is a manifold-valued Savitzky-Golay-style baseline.  For each
    time index, visible rotations in a centered local window are mapped to the
    tangent space of the nearest visible rotation.  A weighted local polynomial
    is fitted against time offsets, the intercept is evaluated at the current
    time, and the result is mapped back to SO(3).

    Rotations are represented as 3-by-3 rotation matrices.  The optional
    ``mask`` marks reliable observations; masked samples are skipped rather than
    treated as identity rotations.  ``smooth_product`` applies the same
    per-component smoother to product states with shape ``(n, k, 3, 3)``.
    """

    def __init__(self, window_size: int = 7, polynomial_degree: int = 2):
        self.window_size = self._validate_window_size(window_size)
        self.polynomial_degree = self._validate_polynomial_degree(polynomial_degree)

    @staticmethod
    def _validate_window_size(window_size: int) -> int:
        try:
            window_size_int = int(window_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("window_size must be a positive odd integer.") from exc
        if window_size_int < 1 or window_size_int % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")
        return window_size_int

    @staticmethod
    def _validate_polynomial_degree(polynomial_degree: int) -> int:
        try:
            degree_int = int(polynomial_degree)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "polynomial_degree must be a non-negative integer."
            ) from exc
        if degree_int < 0:
            raise ValueError("polynomial_degree must be a non-negative integer.")
        return degree_int

    @staticmethod
    def _as_rotation_list(rotations) -> list:
        return SO3ChordalMeanSmoother._as_rotation_list(rotations)

    @staticmethod
    def _identity_rotation():
        return eye(3)

    @staticmethod
    def _normalize_mask(mask, sequence_length: int) -> list[bool]:
        if mask is None:
            return [True for _ in range(sequence_length)]
        mask_array = asarray(mask).reshape(-1)
        if mask_array.shape[0] != sequence_length:
            raise ValueError("mask must have one entry per rotation.")
        return [bool(mask_array[idx]) for idx in range(sequence_length)]

    @staticmethod
    def _window_bounds(sequence_length: int, window_size: int, idx: int):
        half_window = window_size // 2
        return max(0, idx - half_window), min(sequence_length, idx + half_window + 1)

    @staticmethod
    def _nearest_visible_index(mask: Sequence[bool], target: int) -> int | None:
        visible_indices = [idx for idx, is_visible in enumerate(mask) if is_visible]
        if not visible_indices:
            return None
        return min(visible_indices, key=lambda idx: abs(idx - target))

    @staticmethod
    def _skew(vector):
        vector = asarray(vector).reshape(3)
        return array(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ]
        )

    @staticmethod
    def _unskew(matrix):
        matrix = asarray(matrix)
        return array(
            [
                0.5 * (matrix[2, 1] - matrix[1, 2]),
                0.5 * (matrix[0, 2] - matrix[2, 0]),
                0.5 * (matrix[1, 0] - matrix[0, 1]),
            ]
        )

    @staticmethod
    def _project_to_so3(matrix):
        return SO3ChordalMeanSmoother.project_to_so3(matrix)

    @classmethod
    def _exp_map(cls, tangent_vector):
        tangent_vector = asarray(tangent_vector).reshape(3)
        angle = float(linalg.norm(tangent_vector))
        angle_squared = angle * angle
        skew_matrix = cls._skew(tangent_vector)
        if angle < 1e-6:
            first_scale = (
                1.0 - angle_squared / 6.0 + angle_squared * angle_squared / 120.0
            )
            second_scale = (
                0.5 - angle_squared / 24.0 + angle_squared * angle_squared / 720.0
            )
        else:
            first_scale = math.sin(angle) / angle
            second_scale = (1.0 - math.cos(angle)) / angle_squared
        return (
            cls._identity_rotation()
            + first_scale * skew_matrix
            + second_scale * (skew_matrix @ skew_matrix)
        )

    @classmethod
    def _log_map(cls, rotation):  # pylint: disable=too-many-branches
        rotation = cls._project_to_so3(rotation)
        trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
        cosine_angle = min(max(0.5 * (trace - 1.0), -1.0), 1.0)
        angle = math.acos(cosine_angle)
        vee = cls._unskew(rotation - rotation.T)

        if angle < 1e-6:
            return 0.5 * vee
        if math.pi - angle > 1e-5:
            return angle / (2.0 * math.sin(angle)) * vee

        diagonal = [float(rotation[idx, idx]) for idx in range(3)]
        axis_index = max(range(3), key=lambda idx: diagonal[idx])
        axis = [0.0, 0.0, 0.0]
        axis[axis_index] = math.sqrt(max(0.5 * (diagonal[axis_index] + 1.0), 0.0))
        denominator = max(2.0 * axis[axis_index], 1e-8)
        if axis_index == 0:
            axis[1] = float(rotation[0, 1] + rotation[1, 0]) / denominator
            axis[2] = float(rotation[0, 2] + rotation[2, 0]) / denominator
        elif axis_index == 1:
            axis[0] = float(rotation[0, 1] + rotation[1, 0]) / denominator
            axis[2] = float(rotation[1, 2] + rotation[2, 1]) / denominator
        else:
            axis[0] = float(rotation[0, 2] + rotation[2, 0]) / denominator
            axis[1] = float(rotation[1, 2] + rotation[2, 1]) / denominator

        axis = array(axis)
        axis_norm = float(linalg.norm(axis))
        if axis_norm < 1e-8:
            axis = array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm
        return angle * axis

    @classmethod
    def _left_delta(cls, current_rotation, next_rotation):
        return cls._log_map(asarray(next_rotation) @ asarray(current_rotation).T)

    @classmethod
    def _left_apply_delta(cls, tangent_delta, rotation):
        return cls._exp_map(tangent_delta) @ asarray(rotation)

    @staticmethod
    def _local_polynomial_at_zero(offsets: Sequence[float], values, degree: int):
        if len(offsets) == 0:
            raise ValueError("at least one local sample is required.")
        active_degree = min(int(degree), len(offsets) - 1)
        if active_degree == 0:
            nearest_index = min(range(len(offsets)), key=lambda idx: abs(offsets[idx]))
            return asarray(values)[nearest_index]

        scale = max(max(abs(float(offset)) for offset in offsets), 1.0)
        scaled_offsets = [float(offset) / scale for offset in offsets]
        design = array(
            [
                [scaled_offset**power for power in range(active_degree + 1)]
                for scaled_offset in scaled_offsets
            ]
        )
        kernel_weights = array(
            [
                max((1.0 - min(abs(scaled_offset), 1.0) ** 3) ** 3, 1e-6)
                for scaled_offset in scaled_offsets
            ]
        )
        sqrt_weights = reshape(kernel_weights**0.5, (len(offsets), 1))
        weighted_design = design * sqrt_weights
        weighted_values = asarray(values) * sqrt_weights
        normal_matrix = weighted_design.T @ weighted_design
        normal_rhs = weighted_design.T @ weighted_values
        coefficients = linalg.solve(normal_matrix, normal_rhs)
        return coefficients[0]

    def _active_parameters(self, window_size, polynomial_degree):
        active_window_size = (
            self.window_size
            if window_size is None
            else self._validate_window_size(window_size)
        )
        active_degree = (
            self.polynomial_degree
            if polynomial_degree is None
            else self._validate_polynomial_degree(polynomial_degree)
        )
        return active_window_size, active_degree

    def smooth(
        self,
        rotations: Sequence,
        mask=None,
        window_size: int | None = None,
        polynomial_degree: int | None = None,
    ) -> list:
        """Smooth a sequence of SO(3) rotation matrices.

        Parameters
        ----------
        rotations
            Rotation matrix sequence as a Python sequence, ``(n, 3, 3)`` array,
            or ``(3, 3, n)`` array.
        mask
            Optional Boolean reliability mask with one entry per rotation.
            Masked-out rotations are ignored in local polynomial fits.
        window_size
            Optional per-call override for the odd local window size.
        polynomial_degree
            Optional per-call override for the polynomial degree.

        Returns
        -------
        list
            Smoothed SO(3) rotations, one per input rotation.
        """
        rotation_list = self._as_rotation_list(rotations)
        if len(rotation_list) == 0:
            return []

        mask_list = self._normalize_mask(mask, len(rotation_list))
        if not any(mask_list):
            raise ValueError("mask must mark at least one rotation as visible.")

        active_window_size, active_degree = self._active_parameters(
            window_size, polynomial_degree
        )
        smoothed = []
        previous_rotation = self._identity_rotation()
        for idx in range(len(rotation_list)):
            start, stop = self._window_bounds(
                len(rotation_list), active_window_size, idx
            )
            local_indices = [
                local_idx for local_idx in range(start, stop) if mask_list[local_idx]
            ]
            nearest_visible_index = self._nearest_visible_index(mask_list, idx)
            if nearest_visible_index is None:
                smoothed.append(previous_rotation)
                continue
            if not local_indices:
                previous_rotation = rotation_list[nearest_visible_index]
                smoothed.append(previous_rotation)
                continue

            anchor = rotation_list[nearest_visible_index]
            offsets = [float(local_idx - idx) for local_idx in local_indices]
            tangent_vectors = stack(
                [
                    self._left_delta(anchor, rotation_list[local_idx])
                    for local_idx in local_indices
                ],
                axis=0,
            )
            tangent_at_time = self._local_polynomial_at_zero(
                offsets, tangent_vectors, active_degree
            )
            previous_rotation = self._left_apply_delta(tangent_at_time, anchor)
            smoothed.append(previous_rotation)

        return smoothed

    def smooth_product(
        self,
        rotations,
        mask=None,
        window_size: int | None = None,
        polynomial_degree: int | None = None,
    ):
        """Smooth product-state rotations with shape ``(n, k, 3, 3)``.

        The optional mask must have shape ``(n, k)`` and is applied independently
        to each SO(3) component.
        """
        rotation_array = asarray(rotations)
        if ndim(rotation_array) != 4 or rotation_array.shape[-2:] != (3, 3):
            raise ValueError("rotations must have shape (n, k, 3, 3).")

        sequence_length, num_rotations = rotation_array.shape[:2]
        if mask is None:
            mask_array = None
        else:
            mask_array = asarray(mask)
            if mask_array.shape != (sequence_length, num_rotations):
                raise ValueError("mask must have shape (n, k).")

        component_results = []
        for component_idx in range(num_rotations):
            component_mask = (
                None
                if mask_array is None
                else [
                    bool(mask_array[time_idx, component_idx])
                    for time_idx in range(sequence_length)
                ]
            )
            component_results.append(
                stack(
                    self.smooth(
                        [
                            rotation_array[time_idx, component_idx]
                            for time_idx in range(sequence_length)
                        ],
                        mask=component_mask,
                        window_size=window_size,
                        polynomial_degree=polynomial_degree,
                    ),
                    axis=0,
                )
            )
        if not component_results:
            return zeros((sequence_length, 0, 3, 3))
        return stack(component_results, axis=1)


SO3TSGSmoother = SO3TangentSavitzkyGolaySmoother
