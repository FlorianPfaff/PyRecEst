"""Moment matching for weighted Gaussian state hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_INVALID_FLOAT_ARRAY_KINDS = {"b", "S", "U", "c"}
_INVALID_FLOAT_ARRAY_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
)


@dataclass(frozen=True)
class WeightedGaussianHypothesis:
    """One weighted Gaussian hypothesis."""

    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        mean = _as_float_array(self.mean, "mean").reshape(-1)
        covariance = _as_float_array(self.covariance, "covariance")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean must contain only finite values")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance must match mean dimension")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must contain only finite values")
        log_weight = _as_log_weight(self.log_weight)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _symmetrized(covariance))
        object.__setattr__(self, "log_weight", log_weight)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))


def moment_match_gaussian_hypotheses(
    hypotheses: (
        list[WeightedGaussianHypothesis] | tuple[WeightedGaussianHypothesis, ...]
    ),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return moment-matched mean/covariance and normalized weights."""
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")

    dim = hypotheses[0].mean.size
    if any(hypothesis.mean.size != dim for hypothesis in hypotheses):
        raise ValueError("all hypothesis means must have the same dimension")

    weights = normalize_log_weights(
        [hypothesis.log_weight for hypothesis in hypotheses]
    )
    means = np.stack([hypothesis.mean for hypothesis in hypotheses], axis=0)

    mean = weights @ means
    covariance = np.zeros((mean.size, mean.size), dtype=float)
    for weight, hypothesis in zip(weights, hypotheses):
        diff = hypothesis.mean - mean
        covariance += float(weight) * (hypothesis.covariance + np.outer(diff, diff))
    return mean, _symmetrized(covariance), weights


def normalize_log_weights(log_weights: list[float] | np.ndarray) -> np.ndarray:
    """Normalize log weights to probabilities in a numerically stable way."""
    values = _as_float_array(log_weights, "log_weights").reshape(-1)
    if values.size == 0:
        raise ValueError("log_weights must not be empty")
    if np.any(np.isnan(values)):
        raise ValueError("log_weights must not contain NaN values")

    positive_infinite = np.isposinf(values)
    if np.any(positive_infinite):
        weights = np.zeros(values.size, dtype=float)
        weights[positive_infinite] = 1.0 / np.count_nonzero(positive_infinite)
        return weights

    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return np.full(values.size, 1.0 / values.size)
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.size, 1.0 / values.size)
    return weights / total


def _as_float_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.dtype.kind in _INVALID_FLOAT_ARRAY_KINDS or (
        array.dtype.kind == "O"
        and any(
            isinstance(item, _INVALID_FLOAT_ARRAY_SCALAR_TYPES) for item in array.flat
        )
    ):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        return array.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def _as_log_weight(value: Any) -> float:
    array = _as_float_array(value, "log_weight")
    if array.shape != ():
        raise ValueError("log_weight must be a scalar number")
    log_weight = float(array.item())
    if np.isnan(log_weight):
        raise ValueError("log_weight must not be NaN")
    return log_weight


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    array = _as_float_array(matrix, "matrix")
    return 0.5 * (array + array.T)
