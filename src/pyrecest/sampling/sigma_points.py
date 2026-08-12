"""
Sigma-point sampling schemes for unscented transforms.
"""

import math
from numbers import Integral

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all,
    allclose,
    asarray,
    concatenate,
    float64,
    full,
    isfinite,
    linalg,
    stack,
    transpose,
)

_TEXT_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)
_COMPLEX_SCALAR_TYPES = (complex, np.complexfloating)
_TEMPORAL_SCALAR_TYPES = (np.datetime64, np.timedelta64)


def _has_complex_dtype(value) -> bool:
    """Return whether *value* contains complex-valued array entries."""

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        try:
            dtype_kind = np.dtype(dtype).kind
        except (TypeError, ValueError):
            return "complex" in str(dtype).lower()
        if dtype_kind == "c":
            return True
        if dtype_kind != "O":
            return False

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if value_array.dtype.kind == "c":
        return True
    if value_array.dtype.kind != "O":
        return False
    return any(
        isinstance(item, _COMPLEX_SCALAR_TYPES) for item in value_array.reshape(-1)
    )


def _scalar_item(value, name: str):
    """Return a scalar value while rejecting arrays and booleans."""

    if np.ma.is_masked(value):
        raise ValueError(f"{name} must be a scalar")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a scalar")
    shape = getattr(value, "shape", ())
    if tuple(shape) != ():
        raise ValueError(f"{name} must be a scalar")

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        try:
            dtype_kind = np.dtype(dtype).kind
        except (TypeError, ValueError):
            dtype_kind = None
        if dtype_kind in {"M", "m"}:
            raise ValueError(f"{name} must be a scalar")

    try:
        scalar = value.item() if hasattr(value, "item") else value
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a scalar") from exc
    if isinstance(scalar, (bool, np.bool_) + _TEMPORAL_SCALAR_TYPES):
        raise ValueError(f"{name} must be a scalar")
    return scalar


def _to_python_bool(value) -> bool:
    """Convert a scalar backend boolean result to a Python bool."""

    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _validate_positive_integer(value, name: str) -> int:
    scalar = _scalar_item(value, name)
    if not isinstance(scalar, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(scalar)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_finite_scalar(value, name: str) -> float:
    scalar = _scalar_item(value, name)
    if isinstance(scalar, _TEXT_SCALAR_TYPES):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _merwe_scale(alpha: float, n: int, kappa: float) -> float:
    """Return the scaled sigma-point covariance factor without cancellation."""

    scale = alpha * alpha * (n + kappa)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "alpha and kappa must yield a finite, positive sigma-point scale"
        )
    return scale


def _validate_sigma_inputs(x, P, n: int):
    if _has_complex_dtype(x):
        raise ValueError("x must contain real values")
    x = asarray(x, dtype=float64)
    if x.shape != (n,):
        raise ValueError(f"x must have shape ({n},)")
    if not _to_python_bool(all(isfinite(x))):
        raise ValueError("x must contain only finite values")
    if _has_complex_dtype(P):
        raise ValueError("P must contain real values")
    P = asarray(P, dtype=float64)
    if P.shape != (n, n):
        raise ValueError(f"P must have shape ({n}, {n})")
    if not _to_python_bool(all(isfinite(P))):
        raise ValueError("P must contain only finite values")
    if not _to_python_bool(allclose(P, transpose(P), rtol=1e-7, atol=1e-9)):
        raise ValueError("P must be symmetric")
    return x, P


class MerweScaledSigmaPoints:
    """Merwe scaled sigma points (van der Merwe, 2004).

    Parameters
    ----------
    n:
        State dimension.
    alpha:
        Spread of sigma points around the mean (typically 1e-3).
    beta:
        Prior knowledge of the distribution (2 is optimal for Gaussians).
    kappa:
        Secondary scaling parameter (typically 0).
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float):
        self.n = _validate_positive_integer(n, "n")
        self.alpha = _validate_finite_scalar(alpha, "alpha")
        self.beta = _validate_finite_scalar(beta, "beta")
        self.kappa = _validate_finite_scalar(kappa, "kappa")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.n + self.kappa <= 0.0:
            raise ValueError("n + kappa must be positive")
        self._compute_weights()

    def _compute_weights(self):
        n = self.n
        scale = _merwe_scale(self.alpha, n, self.kappa)
        central_mean_weight = (scale - n) / scale
        off_center_weight = 0.5 / scale
        off_center_sum = 2.0 * n * off_center_weight
        mean_weight_sum = math.fsum((central_mean_weight, off_center_sum))

        if (
            not math.isfinite(central_mean_weight)
            or not math.isfinite(off_center_weight)
            or not math.isfinite(off_center_sum)
            or not math.isclose(
                mean_weight_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=8.0 * np.finfo(np.float64).eps,
            )
        ):
            raise ValueError(
                "alpha and kappa must yield finite, normalized mean weights"
            )

        central_covariance_weight = central_mean_weight + (
            1.0 - self.alpha * self.alpha + self.beta
        )
        if not math.isfinite(central_covariance_weight):
            raise ValueError(
                "alpha, beta, and kappa must yield finite covariance weights"
            )

        self.Wm = concatenate(
            [
                asarray([central_mean_weight], dtype=float64),
                full(2 * n, off_center_weight, dtype=float64),
            ]
        )
        self.Wc = concatenate(
            [
                asarray([central_covariance_weight], dtype=float64),
                full(2 * n, off_center_weight, dtype=float64),
            ]
        )

    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        scale = _merwe_scale(self.alpha, n, self.kappa)

        x, P = _validate_sigma_inputs(x, P, n)

        U = linalg.cholesky(scale * P)  # lower-triangular

        positive = [x + U[:, i] for i in range(n)]
        negative = [x - U[:, i] for i in range(n)]
        return stack([x, *positive, *negative])


class JulierSigmaPoints:
    """Julier sigma points (Julier and Uhlmann, 1997).

    Parameters
    ----------
    n:
        State dimension.
    kappa:
        Scaling parameter (``n + kappa`` should be non-zero).
    """

    def __init__(self, n: int, kappa: float = 0.0):
        self.n = _validate_positive_integer(n, "n")
        self.kappa = _validate_finite_scalar(kappa, "kappa")
        if self.n + self.kappa <= 0.0:
            raise ValueError("n + kappa must be positive")
        self._compute_weights()

    def _compute_weights(self):
        n = self.n
        k = n + self.kappa

        self.Wm = concatenate(
            [
                asarray([self.kappa / k], dtype=float64),
                full(2 * n, 0.5 / k, dtype=float64),
            ]
        )
        self.Wc = concatenate(
            [
                asarray([self.kappa / k], dtype=float64),
                full(2 * n, 0.5 / k, dtype=float64),
            ]
        )

    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        k = n + self.kappa

        x, P = _validate_sigma_inputs(x, P, n)

        U = linalg.cholesky(k * P)  # lower-triangular

        positive = [x + U[:, i] for i in range(n)]
        negative = [x - U[:, i] for i in range(n)]
        return stack([x, *positive, *negative])
