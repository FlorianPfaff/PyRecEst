"""Numerical-contract helpers for covariance-like matrices."""

from __future__ import annotations

import numpy as np
from pyrecest.exceptions import (
    DimensionMismatchError,
    NumericalStabilityError,
    ShapeError,
)


def _to_numpy_array(value) -> np.ndarray:
    try:
        import pyrecest.backend as backend

        return np.asarray(backend.to_numpy(value), dtype=float)
    except (
        Exception
    ):  # pragma: no cover - fallback for source-tree bootstrap or unusual array objects
        return np.asarray(value, dtype=float)


def _from_numpy_array(value: np.ndarray):
    try:
        import pyrecest.backend as backend

        return backend.array(value)
    except Exception:  # pragma: no cover
        return value


def _validate_nonnegative_finite(name: str, value: float) -> float:
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar number.") from exc
    if value_array.shape != () or np.issubdtype(value_array.dtype, np.bool_):
        raise ValueError(f"{name} must be a scalar number.")
    try:
        value = float(value_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar number.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _validate_positive_finite(name: str, value: float) -> float:
    value = _validate_nonnegative_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _validate_nonnegative_integer(name: str, value: int) -> int:
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if value_array.shape != () or not np.issubdtype(value_array.dtype, np.integer):
        raise ValueError(f"{name} must be a nonnegative integer.")
    if np.issubdtype(value_array.dtype, np.bool_):
        raise ValueError(f"{name} must be a nonnegative integer.")
    value = int(value_array.item())
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _validate_optional_dimension(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _validate_nonnegative_integer(name, value)


def _is_finite_matrix(matrix: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(matrix)))


def _raise_if_nonfinite_matrix(matrix: np.ndarray, name: str) -> None:
    if not _is_finite_matrix(matrix):
        raise NumericalStabilityError(f"{name} must contain only finite values.")


def _raise_if_not_square_matrix(matrix: np.ndarray, name: str = "matrix") -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ShapeError(f"Expected a square matrix, got shape {matrix.shape}.")


def symmetrize_matrix(matrix):
    """Return ``0.5 * (matrix + matrix.T)`` in the active backend representation."""
    arr = _to_numpy_array(matrix)
    _raise_if_not_square_matrix(arr)
    return _from_numpy_array(0.5 * (arr + arr.T))


def is_symmetric(matrix, *, atol: float = 1e-10) -> bool:
    """Return whether a matrix is symmetric within an absolute tolerance."""
    atol = _validate_nonnegative_finite("atol", atol)
    arr = _to_numpy_array(matrix)
    return bool(
        arr.ndim == 2
        and arr.shape[0] == arr.shape[1]
        and _is_finite_matrix(arr)
        and np.allclose(arr, arr.T, atol=atol, rtol=0.0)
    )


def is_positive_semidefinite(matrix, *, atol: float = 1e-10) -> bool:
    """Return whether a symmetric matrix is positive semidefinite within tolerance."""
    atol = _validate_nonnegative_finite("atol", atol)
    arr = _to_numpy_array(matrix)
    if (
        arr.ndim != 2
        or arr.shape[0] != arr.shape[1]
        or not _is_finite_matrix(arr)
        or not is_symmetric(arr, atol=atol)
    ):
        return False
    if arr.shape[0] == 0:
        return True
    return bool(np.min(np.linalg.eigvalsh(arr)) >= -atol)


def nearest_symmetric_psd(matrix, *, min_eigenvalue: float = 0.0):
    """Project a symmetric matrix to the nearest eigenvalue-clipped PSD matrix.

    This helper is intended for diagnostics and controlled numerical repair. It
    should not silently replace validation in algorithms where invalid covariance
    matrices indicate a modeling error.
    """
    min_eigenvalue = _validate_nonnegative_finite("min_eigenvalue", min_eigenvalue)

    arr = _to_numpy_array(matrix)
    _raise_if_not_square_matrix(arr)
    _raise_if_nonfinite_matrix(arr, "matrix")
    sym = 0.5 * (arr + arr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, min_eigenvalue)
    repaired = (eigvecs * clipped) @ eigvecs.T
    return _from_numpy_array(0.5 * (repaired + repaired.T))


def jittered_cholesky(matrix, *, initial_jitter: float = 1e-12, max_attempts: int = 8):
    """Return a Cholesky factor and the jitter used to obtain it.

    The function tries the raw matrix first, then repeatedly adds diagonal
    jitter. It raises :class:`NumericalStabilityError` if no factorization is
    found within ``max_attempts``.
    """
    initial_jitter = _validate_positive_finite("initial_jitter", initial_jitter)
    max_attempts = _validate_nonnegative_integer("max_attempts", max_attempts)

    arr = _to_numpy_array(matrix)
    _raise_if_not_square_matrix(arr)
    _raise_if_nonfinite_matrix(arr, "matrix")
    sym = 0.5 * (arr + arr.T)
    eye = np.eye(sym.shape[0])
    jitter = 0.0
    for attempt in range(max_attempts + 1):
        try:
            factor = np.linalg.cholesky(sym + jitter * eye)
            return _from_numpy_array(factor), jitter
        except np.linalg.LinAlgError:
            jitter = initial_jitter if attempt == 0 else jitter * 10.0
    raise NumericalStabilityError(
        f"Cholesky factorization failed after {max_attempts} jitter attempts."
    )


def assert_covariance_matrix(
    matrix, *, name: str = "covariance", dim: int | None = None, atol: float = 1e-10
):
    """Validate a covariance matrix and return it in the active backend representation."""
    atol = _validate_nonnegative_finite("atol", atol)
    dim = _validate_optional_dimension("dim", dim)
    arr = _to_numpy_array(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ShapeError(f"{name} must be a square matrix, got shape {arr.shape}.")
    if dim is not None and arr.shape[0] != dim:
        raise DimensionMismatchError(
            f"{name} dimension {arr.shape[0]} does not match expected dimension {dim}."
        )
    _raise_if_nonfinite_matrix(arr, name)
    if not is_symmetric(arr, atol=atol):
        raise NumericalStabilityError(f"{name} must be symmetric within atol={atol}.")
    if not is_positive_semidefinite(arr, atol=atol):
        raise NumericalStabilityError(
            f"{name} must be positive semidefinite within atol={atol}."
        )
    return _from_numpy_array(arr)


__all__ = [
    "assert_covariance_matrix",
    "is_positive_semidefinite",
    "is_symmetric",
    "jittered_cholesky",
    "nearest_symmetric_psd",
    "symmetrize_matrix",
]
