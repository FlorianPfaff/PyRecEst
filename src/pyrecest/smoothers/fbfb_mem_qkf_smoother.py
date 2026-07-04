"""Forward-backward MEM-QKF smoother public wrapper fixes."""

from __future__ import annotations

from typing import Any

from pyrecest.backend import asarray
from pyrecest.backend import copy as backend_copy

from .fixed_lag_mem_qkf_smoother import (
    ForwardBackwardForwardBackwardMEMQKFSmoother as _BaseForwardBackwardForwardBackwardMEMQKFSmoother,
)


class ForwardBackwardForwardBackwardMEMQKFSmoother(
    _BaseForwardBackwardForwardBackwardMEMQKFSmoother
):
    """FBFB MEM-QKF smoother with NumPy-style matrix sequence normalization."""

    @staticmethod
    def _normalize_optional_matrix_sequence(
        values, length: int, name: str
    ) -> list[Any | None]:
        """Normalize optional per-scan matrices.

        Plain Python nested-list matrices, such as ``[[r11, r12], [r21, r22]]``,
        are single matrices and must be repeated for every scan.  Only fall back
        to list/tuple sequence handling after ruling out matrix-shaped array-like
        inputs so rows of a matrix are not misinterpreted as per-scan entries.
        """
        if values is None:
            return [None] * length

        try:
            values_array = asarray(values)
        except (TypeError, ValueError):
            values_array = None
        if values_array is not None:
            if values_array.ndim == 2:
                return [backend_copy(values_array) for _ in range(length)]
            if values_array.ndim == 3 and values_array.shape[0] == length:
                return [values_array[idx] for idx in range(length)]

        if isinstance(values, (list, tuple)):
            if len(values) != length:
                raise ValueError(f"{name} must have length {length}.")
            return [None if value is None else asarray(value) for value in values]

        raise ValueError(f"{name} must be a matrix or a sequence of matrices.")


FBFBMEMQKFSmoother = ForwardBackwardForwardBackwardMEMQKFSmoother
ForwardBackwardMEMQKFSmoother = ForwardBackwardForwardBackwardMEMQKFSmoother
