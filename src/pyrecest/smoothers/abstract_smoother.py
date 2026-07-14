"""Abstract base class for all smoothers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pyrecest.backend import asarray, copy, ndim
from pyrecest.distributions import GaussianDistribution


class AbstractSmoother(ABC):
    """Abstract base class for all smoothers."""

    @staticmethod
    def _as_gaussian(state: "GaussianDistribution | tuple") -> GaussianDistribution:
        if isinstance(state, GaussianDistribution):
            return state
        if isinstance(state, tuple) and len(state) == 2:
            return GaussianDistribution(state[0], state[1], check_validity=False)
        raise ValueError(
            "State must be a GaussianDistribution or a tuple of (mean, covariance)."
        )

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _normalize_measurements(measurements) -> list:
        if isinstance(measurements, (list, tuple)):
            normalized = [
                asarray(measurement).reshape(-1) for measurement in measurements
            ]
            if normalized and any(
                measurement.shape != normalized[0].shape
                for measurement in normalized[1:]
            ):
                raise ValueError("All measurements must have the same dimension.")
            return normalized

        measurements_array = asarray(measurements)
        if ndim(measurements_array) == 0:
            return [measurements_array.reshape((1,))]
        if ndim(measurements_array) == 1:
            return [asarray([measurement]) for measurement in measurements_array]
        if ndim(measurements_array) == 2:
            return [
                measurements_array[idx] for idx in range(measurements_array.shape[0])
            ]
        raise ValueError(
            "Measurements must be a 1-D or 2-D array, or a Python sequence."
        )

    @staticmethod
    def _normalize_matrix_sequence(  # pylint: disable=too-many-return-statements,too-many-branches
        values, length: int, name: str, matrix_dim: int, default=None
    ) -> list:
        if length == 0:
            return []

        if values is None:
            if default is None:
                raise ValueError(f"{name} must be provided.")
            default_arr = asarray(default)
            return [copy(default_arr) for _ in range(length)]

        try:
            values_arr = asarray(values)
        except (TypeError, ValueError):
            values_arr = None
        if values_arr is not None:
            if ndim(values_arr) == 0:
                if matrix_dim != 1:
                    raise ValueError(
                        f"Scalar input for {name} is only supported in one-dimensional models."
                    )
                scalar_matrix = asarray([[values_arr]])
                return [copy(scalar_matrix) for _ in range(length)]
            if (
                ndim(values_arr) == 1
                and matrix_dim == 1
                and values_arr.shape[0] == length
            ):
                return [asarray([[values_arr[idx]]]) for idx in range(length)]
            if ndim(values_arr) == 2:
                return [copy(values_arr) for _ in range(length)]
            if ndim(values_arr) == 3 and values_arr.shape[0] == length:
                return [copy(values_arr[idx]) for idx in range(length)]

        if isinstance(values, (list, tuple)) and len(values) == length:
            normalized_values = []
            for value in values:
                value_arr = asarray(value)
                if ndim(value_arr) == 0:
                    if matrix_dim != 1:
                        raise ValueError(
                            f"Scalar entries in {name} are only supported in one-dimensional models."
                        )
                    normalized_values.append(asarray([[value_arr]]))
                else:
                    normalized_values.append(value_arr)
            return normalized_values

        raise ValueError(
            f"{name} must be a single matrix or a sequence with length {length}."
        )

    @staticmethod
    def _normalize_vector_sequence(  # pylint: disable=too-many-return-statements
        values, length: int, name: str, vector_dim: int
    ) -> list:
        if length == 0:
            return []

        if values is None:
            return [None] * length

        expected_shape = (vector_dim,)
        shape_error = f"{name} must contain vectors with shape {expected_shape}."

        try:
            values_arr = asarray(values)
        except (TypeError, ValueError, RuntimeError):
            values_arr = None
        if values_arr is not None:
            if ndim(values_arr) == 0:
                if vector_dim != 1:
                    raise ValueError(
                        f"Scalar input for {name} is only supported in one-dimensional models."
                    )
                scalar_vector = asarray([values_arr])
                return [copy(scalar_vector) for _ in range(length)]
            if ndim(values_arr) == 1:
                if vector_dim == 1 and values_arr.shape[0] == length:
                    return [asarray([values_arr[idx]]) for idx in range(length)]
                if values_arr.shape[0] != vector_dim:
                    raise ValueError(shape_error)
                return [copy(values_arr) for _ in range(length)]
            if ndim(values_arr) == 2 and values_arr.shape[0] == length:
                if values_arr.shape[1] != vector_dim:
                    raise ValueError(shape_error)
                return [copy(values_arr[idx]) for idx in range(length)]

        if isinstance(values, (list, tuple)) and len(values) == length:
            normalized_values = []
            for value in values:
                if value is None:
                    normalized_values.append(None)
                    continue
                value_arr = asarray(value)
                if ndim(value_arr) == 0:
                    if vector_dim != 1:
                        raise ValueError(
                            f"Scalar entries in {name} are only supported in one-dimensional models."
                        )
                    normalized_values.append(asarray([value_arr]))
                elif ndim(value_arr) != 1 or value_arr.shape[0] != vector_dim:
                    raise ValueError(shape_error)
                else:
                    normalized_values.append(value_arr)
            return normalized_values

        raise ValueError(
            f"{name} must be a single vector or a sequence with length {length}."
        )

    @abstractmethod
    def smooth(self, *args, **kwargs):
        """Smooth a sequence of states produced by a forward pass."""
