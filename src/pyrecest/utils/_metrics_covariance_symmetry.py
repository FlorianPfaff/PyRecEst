"""Runtime covariance validation for utility metrics."""

from __future__ import annotations

import numpy as np


def patch_metrics_covariance_symmetry() -> None:
    """Reject materially asymmetric covariance inputs in Wasserstein metrics."""

    from . import metrics as metrics_module  # pylint: disable=import-outside-toplevel

    original = metrics_module._as_covariance_matrix  # pylint: disable=protected-access
    if getattr(original, "_pyrecest_symmetry_validation", False):
        return

    def _as_covariance_matrix(value, name):
        matrix = metrics_module._as_numeric_array(  # pylint: disable=protected-access
            value,
            name,
        )
        metrics_module._validate_square_matrix(  # pylint: disable=protected-access
            matrix,
            name,
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values")

        transpose = matrix.T
        scale = np.maximum(np.maximum(np.abs(matrix), np.abs(transpose)), 1.0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            relative_asymmetry = np.abs(matrix / scale - transpose / scale)
        if np.any(relative_asymmetry > 1e-12):
            raise ValueError(f"{name} must be symmetric")

        matrix = metrics_module._symmetrize(  # pylint: disable=protected-access
            matrix
        )
        metrics_module._validate_positive_semidefinite(  # pylint: disable=protected-access
            matrix,
            name,
        )
        return matrix

    _as_covariance_matrix.__name__ = getattr(
        original,
        "__name__",
        "_as_covariance_matrix",
    )
    _as_covariance_matrix.__doc__ = getattr(original, "__doc__", None)
    _as_covariance_matrix.__module__ = getattr(original, "__module__", __name__)
    _as_covariance_matrix._pyrecest_symmetry_validation = True
    metrics_module._as_covariance_matrix = (  # pylint: disable=protected-access
        _as_covariance_matrix
    )
