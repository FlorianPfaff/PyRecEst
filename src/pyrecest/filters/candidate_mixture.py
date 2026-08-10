"""Robust finite Gaussian-mixture measurement factors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gaussian_hypothesis_mixture import normalize_log_weights

__all__ = [
    "GaussianMixtureMeasurementEvaluation",
    "GaussianMixtureMeasurementFactor",
]

LOSS_CHOICES = ("squared", "huber")
_INVALID_ARRAY_KINDS = {"b", "S", "U", "c", "M", "m"}
_INVALID_OBJECT_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
)


@dataclass(frozen=True)
class GaussianMixtureMeasurementEvaluation:
    """Responsibilities, moments, and diagnostics for one factor evaluation."""

    predicted_measurement: np.ndarray
    residuals: np.ndarray
    mahalanobis_distances: np.ndarray
    robust_costs: np.ndarray
    component_log_weights: np.ndarray
    responsibilities: np.ndarray
    log_evidence: float
    mean: np.ndarray
    covariance: np.ndarray
    entropy: float
    effective_component_count: float
    dominant_index: int


@dataclass(frozen=True)
class GaussianMixtureMeasurementFactor:
    """A finite Gaussian-mixture factor with an optional linear observation model.

    For component ``i``, the unnormalized log potential is

    ``log_weights[i] - rho(r_i) - 0.5*w*log(det(R_i))``,

    where ``r_i`` is its Mahalanobis residual norm, ``rho`` is squared or Huber
    loss, and ``w`` is ``log_determinant_weight``. With squared loss and ``w=1``
    this is the Gaussian-mixture log likelihood up to the common
    ``-measurement_dim/2*log(2*pi)`` term.
    """

    means: np.ndarray
    covariances: np.ndarray
    log_weights: np.ndarray | None = None
    observation_matrix: np.ndarray | None = None
    offset: np.ndarray | None = None
    loss: str = "squared"
    huber_delta: float = 1.0
    log_determinant_weight: float = 1.0
    _cholesky: np.ndarray = field(init=False, repr=False, compare=False)
    _log_determinants: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        means = _real_array(self.means, "means")
        if means.ndim != 2 or 0 in means.shape:
            raise ValueError("means must have shape (component_count, measurement_dim)")
        if not np.all(np.isfinite(means)):
            raise ValueError("means must contain only finite values")
        means = means.copy()
        component_count, measurement_dim = means.shape

        covariances = _real_array(self.covariances, "covariances")
        if covariances.ndim == 2:
            if covariances.shape != (measurement_dim, measurement_dim):
                raise ValueError("covariances must match the measurement dimension")
            covariances = np.broadcast_to(
                covariances, (component_count, measurement_dim, measurement_dim)
            ).copy()
        elif covariances.shape != (component_count, measurement_dim, measurement_dim):
            raise ValueError(
                "covariances must have shape (measurement_dim, measurement_dim) "
                "or (component_count, measurement_dim, measurement_dim)"
            )
        else:
            covariances = covariances.copy()
        if not np.all(np.isfinite(covariances)):
            raise ValueError("covariances must contain only finite values")
        covariances = np.stack([_symmetrize(value) for value in covariances])
        try:
            cholesky = np.linalg.cholesky(covariances)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariances must be positive definite") from exc
        log_determinants = 2.0 * np.sum(
            np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
        )

        if self.log_weights is None:
            log_weights = np.zeros(component_count)
        else:
            log_weights = _real_array(self.log_weights, "log_weights")
            if log_weights.ndim == 0 and component_count == 1:
                log_weights = log_weights.reshape(1)
            if log_weights.shape != (component_count,):
                raise ValueError("log_weights must contain one value per component")
            if np.any(np.isnan(log_weights)):
                raise ValueError("log_weights must not contain NaN values")
            if not np.any(log_weights > -np.inf):
                raise ValueError("log_weights must contain positive total mass")
            log_weights = log_weights.copy()

        observation_matrix = self.observation_matrix
        if observation_matrix is not None:
            observation_matrix = _real_array(observation_matrix, "observation_matrix")
            if (
                observation_matrix.ndim != 2
                or observation_matrix.shape[0] != measurement_dim
                or observation_matrix.shape[1] == 0
            ):
                raise ValueError(
                    "observation_matrix must have shape (measurement_dim, state_dim)"
                )
            if not np.all(np.isfinite(observation_matrix)):
                raise ValueError("observation_matrix must contain only finite values")
            observation_matrix = observation_matrix.copy()

        offset = (
            np.zeros(measurement_dim)
            if self.offset is None
            else _real_array(self.offset, "offset")
        )
        if offset.ndim == 0 and measurement_dim == 1:
            offset = offset.reshape(1)
        if offset.shape != (measurement_dim,):
            raise ValueError("offset must match the measurement dimension")
        if not np.all(np.isfinite(offset)):
            raise ValueError("offset must contain only finite values")

        if self.loss not in LOSS_CHOICES:
            raise ValueError(f"unsupported loss {self.loss!r}")
        huber_delta = _finite_scalar(self.huber_delta, "huber_delta")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        determinant_weight = _finite_scalar(
            self.log_determinant_weight, "log_determinant_weight"
        )

        object.__setattr__(self, "means", _readonly(means))
        object.__setattr__(self, "covariances", _readonly(covariances))
        object.__setattr__(self, "log_weights", _readonly(log_weights))
        object.__setattr__(
            self,
            "observation_matrix",
            None if observation_matrix is None else _readonly(observation_matrix),
        )
        object.__setattr__(self, "offset", _readonly(offset))
        object.__setattr__(self, "huber_delta", huber_delta)
        object.__setattr__(self, "log_determinant_weight", determinant_weight)
        object.__setattr__(self, "_cholesky", _readonly(cholesky))
        object.__setattr__(self, "_log_determinants", _readonly(log_determinants))

    @property
    def component_count(self) -> int:
        """Number of mixture components."""

        return int(self.means.shape[0])

    @property
    def measurement_dimension(self) -> int:
        """Dimension of each candidate measurement."""

        return int(self.means.shape[1])

    @property
    def state_dimension(self) -> int:
        """Required state dimension."""

        return (
            self.measurement_dimension
            if self.observation_matrix is None
            else int(self.observation_matrix.shape[1])
        )

    def evaluate(self, state: np.ndarray) -> GaussianMixtureMeasurementEvaluation:
        """Evaluate the factor at ``state`` and moment-match its components."""

        state = _real_array(state, "state")
        if state.ndim == 0 and self.state_dimension == 1:
            state = state.reshape(1)
        if state.shape != (self.state_dimension,):
            raise ValueError("state must match the factor state dimension")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must contain only finite values")

        predicted = (
            state if self.observation_matrix is None else self.observation_matrix @ state
        ) + self.offset
        residuals = self.means - predicted
        whitened = np.linalg.solve(self._cholesky, residuals[..., None])[..., 0]
        distances = _row_norm(whitened)
        costs = _robust_cost(distances, self.loss, float(self.huber_delta))
        component_log_weights = (
            self.log_weights
            - costs
            - 0.5 * self.log_determinant_weight * self._log_determinants
        )
        responsibilities = normalize_log_weights(component_log_weights)
        mean, covariance = self.moment_match(responsibilities)
        positive = responsibilities > 0.0
        entropy = float(
            -np.sum(responsibilities[positive] * np.log(responsibilities[positive]))
        )
        return GaussianMixtureMeasurementEvaluation(
            predicted_measurement=np.asarray(predicted).copy(),
            residuals=np.asarray(residuals).copy(),
            mahalanobis_distances=distances.copy(),
            robust_costs=costs.copy(),
            component_log_weights=np.asarray(component_log_weights).copy(),
            responsibilities=np.asarray(responsibilities).copy(),
            log_evidence=_logsumexp(component_log_weights),
            mean=mean,
            covariance=covariance,
            entropy=entropy,
            effective_component_count=float(np.exp(entropy)),
            dominant_index=int(np.argmax(responsibilities)),
        )

    def moment_match(self, responsibilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mixture mean and covariance for caller-supplied responsibilities."""

        weights = _probabilities(responsibilities, self.component_count)
        mean = weights @ self.means
        differences = self.means - mean
        covariance = np.sum(
            weights[:, None, None]
            * (self.covariances + differences[:, :, None] * differences[:, None, :]),
            axis=0,
        )
        return np.asarray(mean), _symmetrize(covariance)


def _robust_cost(distance: np.ndarray, loss: str, delta: float) -> np.ndarray:
    if loss == "squared":
        return 0.5 * distance**2
    return np.where(
        distance <= delta,
        0.5 * distance**2,
        delta * (distance - 0.5 * delta),
    )


def _probabilities(values: Any, expected_count: int) -> np.ndarray:
    values = _real_array(values, "responsibilities")
    if values.ndim == 0 and expected_count == 1:
        values = values.reshape(1)
    if values.shape != (expected_count,):
        raise ValueError("responsibilities must contain one value per component")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("responsibilities must be finite and non-negative")
    scale = float(np.max(values))
    if scale <= 0.0:
        raise ValueError("responsibilities must contain positive total mass")
    scaled = values / scale
    total = float(np.sum(scaled))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("responsibilities must contain positive finite total mass")
    return scaled / total


def _row_norm(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.hypot.reduce(np.abs(row), initial=0.0) for row in np.asarray(values)],
        dtype=float,
    )


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(np.isposinf(values)):
        return float("inf")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise ValueError("component log weights must contain positive total mass")
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def _real_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.dtype.kind in _INVALID_ARRAY_KINDS or (
        array.dtype.kind == "O"
        and any(isinstance(item, _INVALID_OBJECT_TYPES) for item in array.flat)
    ):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        return array.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def _finite_scalar(value: Any, name: str) -> float:
    value = _real_array(value, name)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar number")
    scalar = float(value.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    with np.errstate(over="ignore"):
        result = 0.5 * (matrix + matrix.T)
    overflowed = ~np.isfinite(result) & np.isfinite(matrix) & np.isfinite(matrix.T)
    if np.any(overflowed):
        result[overflowed] = 0.5 * matrix[overflowed] + 0.5 * matrix.T[overflowed]
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=float).copy()
    result.setflags(write=False)
    return result
