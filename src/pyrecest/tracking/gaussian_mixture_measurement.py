"""Gaussian-mixture measurement factors for robust candidate association.

The utilities in this module are independent of a particular tracker or tabular
schema. A factor represents several Gaussian candidates for one measurement
occasion, evaluates their posterior responsibilities at a state, and exposes a
moment-matched Gaussian approximation for downstream MAP or filtering code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

MixtureLoss = Literal["huber", "squared"]


@dataclass(frozen=True)
class GaussianMixtureMeasurementEvaluation:
    """Per-component statistics for one state evaluation."""

    responsibilities: NDArray[np.float64]
    log_unnormalized_responsibilities: NDArray[np.float64]
    residuals: NDArray[np.float64]
    residual_norms: NDArray[np.float64]
    mahalanobis_distances: NDArray[np.float64]
    component_costs: NDArray[np.float64]
    entropy: float
    effective_component_count: float
    dominant_index: int


@dataclass(frozen=True)
class GaussianMixtureMomentMatch:
    """Moment-matched measurement-space Gaussian."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    isotropic_variance: float


class GaussianMixtureMeasurementFactor:
    """A Gaussian-mixture measurement likelihood evaluated at a linear state.

    Parameters
    ----------
    component_means:
        Candidate measurement means with shape ``(component_count,
        measurement_dimension)``.
    component_covariances:
        Symmetric positive-definite candidate covariances with shape
        ``(component_count, measurement_dimension, measurement_dimension)``.
    log_weights:
        Optional unnormalized component log priors. ``-inf`` disables a
        component. The values need not sum to one.
    observation_matrix:
        Optional linear map from state space to measurement space. The identity
        map is used when omitted.
    loss:
        ``"squared"`` for the Gaussian quadratic cost or ``"huber"`` for a
        Huber cost applied to each component's Mahalanobis distance.
    huber_delta:
        Positive Huber transition point.
    include_gaussian_normalization:
        Include each covariance determinant and the common ``2*pi`` term in the
        component log likelihood. Leave this disabled when those terms are
        already represented in ``log_weights`` or when intentionally using
        custom uncertainty penalties.
    """

    def __init__(
        self,
        component_means: ArrayLike,
        component_covariances: ArrayLike,
        *,
        log_weights: ArrayLike | None = None,
        observation_matrix: ArrayLike | None = None,
        loss: MixtureLoss = "squared",
        huber_delta: float = 1.0,
        include_gaussian_normalization: bool = False,
    ) -> None:
        means = _as_finite_real_array(
            component_means,
            field="component_means",
            ndim=2,
        )
        if means.shape[0] == 0 or means.shape[1] == 0:
            raise ValueError(
                "component_means must contain at least one nonempty component"
            )
        covariance = _as_finite_real_array(
            component_covariances,
            field="component_covariances",
            ndim=3,
        )
        expected_covariance_shape = (
            means.shape[0],
            means.shape[1],
            means.shape[1],
        )
        if covariance.shape != expected_covariance_shape:
            raise ValueError(
                "component_covariances must have shape "
                f"{expected_covariance_shape}, got {covariance.shape}"
            )
        covariance = np.array(covariance, dtype=float, copy=True)
        cholesky = np.empty_like(covariance)
        for component_index, component_covariance in enumerate(covariance):
            if not np.allclose(
                component_covariance,
                component_covariance.T,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("component_covariances must be symmetric")
            component_covariance = (
                0.5 * component_covariance + 0.5 * component_covariance.T
            )
            try:
                component_cholesky = np.linalg.cholesky(component_covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "component_covariances must be positive definite"
                ) from exc
            covariance[component_index] = component_covariance
            cholesky[component_index] = component_cholesky

        component_count, measurement_dimension = means.shape
        if log_weights is None:
            prior = np.zeros(component_count, dtype=float)
        else:
            prior = _as_real_array(
                log_weights,
                field="log_weights",
                ndim=1,
            )
            if prior.shape != (component_count,):
                raise ValueError(
                    f"log_weights must have shape {(component_count,)}, "
                    f"got {prior.shape}"
                )
            if np.isnan(prior).any() or np.isposinf(prior).any():
                raise ValueError(
                    "log_weights may be finite or -inf, but not NaN or +inf"
                )
            if not np.isfinite(prior).any():
                raise ValueError("at least one log weight must be finite")

        if observation_matrix is None:
            observation = np.eye(measurement_dimension, dtype=float)
        else:
            observation = _as_finite_real_array(
                observation_matrix,
                field="observation_matrix",
                ndim=2,
            )
            if observation.shape[0] != measurement_dimension:
                raise ValueError(
                    "observation_matrix row count must match the measurement "
                    "dimension"
                )
            if observation.shape[1] == 0:
                raise ValueError(
                    "observation_matrix must have at least one state column"
                )

        if loss not in ("huber", "squared"):
            raise ValueError("loss must be 'huber' or 'squared'")
        delta = _finite_scalar(huber_delta, field="huber_delta")
        if delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if not isinstance(include_gaussian_normalization, (bool, np.bool_)):
            raise ValueError(
                "include_gaussian_normalization must be a Boolean scalar"
            )

        self._component_means = np.array(means, dtype=float, copy=True)
        self._component_covariances = covariance
        self._cholesky_factors = cholesky
        self._log_weights = np.array(prior, dtype=float, copy=True)
        self._observation_matrix = np.array(observation, dtype=float, copy=True)
        self._loss = loss
        self._huber_delta = delta
        self._include_gaussian_normalization = bool(
            include_gaussian_normalization
        )

    @classmethod
    def from_isotropic_standard_deviations(
        cls,
        component_means: ArrayLike,
        standard_deviations: ArrayLike,
        *,
        log_weights: ArrayLike | None = None,
        observation_matrix: ArrayLike | None = None,
        loss: MixtureLoss = "squared",
        huber_delta: float = 1.0,
        include_gaussian_normalization: bool = False,
    ) -> "GaussianMixtureMeasurementFactor":
        """Construct a factor with isotropic per-component covariances."""

        means = _as_finite_real_array(
            component_means,
            field="component_means",
            ndim=2,
        )
        sigma = _as_finite_real_array(
            standard_deviations,
            field="standard_deviations",
        )
        if sigma.ndim == 0:
            sigma = np.full(means.shape[0], float(sigma), dtype=float)
        if sigma.ndim != 1 or sigma.shape != (means.shape[0],):
            raise ValueError(
                "standard_deviations must be a scalar or one value per component"
            )
        if np.any(sigma <= 0.0):
            raise ValueError("standard_deviations must be positive")
        identity = np.eye(means.shape[1], dtype=float)
        covariances = sigma[:, None, None] ** 2 * identity[None, :, :]
        return cls(
            means,
            covariances,
            log_weights=log_weights,
            observation_matrix=observation_matrix,
            loss=loss,
            huber_delta=huber_delta,
            include_gaussian_normalization=include_gaussian_normalization,
        )

    @property
    def component_count(self) -> int:
        """Number of mixture components."""

        return int(self._component_means.shape[0])

    @property
    def measurement_dimension(self) -> int:
        """Dimension of each candidate measurement."""

        return int(self._component_means.shape[1])

    @property
    def state_dimension(self) -> int:
        """Dimension expected by :meth:`evaluate`."""

        return int(self._observation_matrix.shape[1])

    def evaluate(
        self,
        state: ArrayLike,
    ) -> GaussianMixtureMeasurementEvaluation:
        """Evaluate component costs and posterior responsibilities at ``state``."""

        state_array = _as_finite_real_array(state, field="state", ndim=1)
        if state_array.shape != (self.state_dimension,):
            raise ValueError(
                f"state must have shape {(self.state_dimension,)}, "
                f"got {state_array.shape}"
            )
        predicted_measurement = self._observation_matrix @ state_array
        residuals = self._component_means - predicted_measurement[None, :]
        whitened = np.empty_like(residuals)
        for component_index in range(self.component_count):
            whitened[component_index] = np.linalg.solve(
                self._cholesky_factors[component_index],
                residuals[component_index],
            )
        mahalanobis = _stable_row_norms(whitened)
        residual_norms = _stable_row_norms(residuals)
        costs = _robust_cost(
            mahalanobis,
            loss=self._loss,
            huber_delta=self._huber_delta,
        )
        log_unnormalized = self._log_weights - costs
        if self._include_gaussian_normalization:
            log_determinants = 2.0 * np.sum(
                np.log(
                    np.diagonal(
                        self._cholesky_factors,
                        axis1=1,
                        axis2=2,
                    )
                ),
                axis=1,
            )
            log_unnormalized = log_unnormalized - 0.5 * (
                self.measurement_dimension * np.log(2.0 * np.pi)
                + log_determinants
            )
        responsibilities = _softmax_log_weights(log_unnormalized)
        entropy = _responsibility_entropy(responsibilities)
        return GaussianMixtureMeasurementEvaluation(
            responsibilities=responsibilities,
            log_unnormalized_responsibilities=np.array(
                log_unnormalized,
                dtype=float,
                copy=True,
            ),
            residuals=np.array(residuals, dtype=float, copy=True),
            residual_norms=residual_norms,
            mahalanobis_distances=mahalanobis,
            component_costs=costs,
            entropy=entropy,
            effective_component_count=float(np.exp(entropy)),
            dominant_index=int(np.argmax(responsibilities)),
        )

    def moment_match(
        self,
        responsibilities: ArrayLike | None = None,
    ) -> GaussianMixtureMomentMatch:
        """Moment-match the component Gaussians using supplied responsibilities."""

        if responsibilities is None:
            finite_log_weights = self._log_weights[
                np.isfinite(self._log_weights)
            ]
            weights = normalize_mixture_responsibilities(
                np.exp(self._log_weights - np.max(finite_log_weights))
            )
        else:
            weights = normalize_mixture_responsibilities(responsibilities)
        if weights.shape != (self.component_count,):
            raise ValueError(
                "responsibilities must contain one value per mixture component"
            )
        mean = np.sum(weights[:, None] * self._component_means, axis=0)
        centered = self._component_means - mean[None, :]
        between = np.einsum("ki,kj->kij", centered, centered)
        covariance = np.sum(
            weights[:, None, None]
            * (self._component_covariances + between),
            axis=0,
        )
        covariance = 0.5 * covariance + 0.5 * covariance.T
        isotropic_variance = float(
            np.trace(covariance) / self.measurement_dimension
        )
        return GaussianMixtureMomentMatch(
            mean=np.array(mean, dtype=float, copy=True),
            covariance=np.array(covariance, dtype=float, copy=True),
            isotropic_variance=isotropic_variance,
        )


def normalize_mixture_responsibilities(
    values: ArrayLike,
) -> NDArray[np.float64]:
    """Return a finite nonnegative vector normalized without overflow."""

    weights = _as_finite_real_array(
        values,
        field="responsibilities",
        ndim=1,
    )
    if weights.size == 0:
        raise ValueError("responsibilities must not be empty")
    if np.any(weights < 0.0):
        raise ValueError("responsibilities must be nonnegative")
    scale = float(np.max(weights))
    if scale <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    scaled = weights / scale
    total = float(np.sum(scaled))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError(
            "responsibilities must have positive finite total mass"
        )
    return np.asarray(scaled / total, dtype=float)


def blend_mixture_responsibilities_with_uniform(
    responsibilities: ArrayLike,
    fraction: float,
) -> NDArray[np.float64]:
    """Blend normalized responsibilities with a uniform distribution."""

    weights = normalize_mixture_responsibilities(responsibilities)
    amount = _finite_scalar(fraction, field="fraction")
    if not 0.0 <= amount <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    return normalize_mixture_responsibilities(
        (1.0 - amount) * weights + amount / weights.size
    )


def balance_mixture_responsibilities(
    responsibilities: ArrayLike,
    labels: ArrayLike,
    balance: float,
) -> NDArray[np.float64]:
    """Blend component mass with equal total mass over distinct labels.

    Within each label group, the original relative component responsibilities
    are retained. ``balance=0`` returns the normalized input, while
    ``balance=1`` assigns equal total mass to every distinct string label.
    """

    weights = normalize_mixture_responsibilities(responsibilities)
    amount = _finite_scalar(balance, field="balance")
    if not 0.0 <= amount <= 1.0:
        raise ValueError("balance must be within [0, 1]")
    if _contains_masked_value(labels):
        raise ValueError("labels must not contain masked values")
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.shape != weights.shape:
        raise ValueError(
            "labels must contain one value per responsibility"
        )
    label_text = label_array.astype(str)
    unique_labels = sorted(set(label_text.tolist()))
    if not unique_labels:
        return weights
    balanced = np.zeros_like(weights)
    label_mass = 1.0 / len(unique_labels)
    for label in unique_labels:
        mask = label_text == label
        balanced[mask] = label_mass * normalize_mixture_responsibilities(
            weights[mask]
        )
    return normalize_mixture_responsibilities(
        (1.0 - amount) * weights + amount * balanced
    )


def _softmax_log_weights(
    log_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    finite = np.isfinite(log_weights)
    if not finite.any():
        raise ValueError(
            "at least one component must have finite posterior log weight"
        )
    maximum = float(np.max(log_weights[finite]))
    shifted = np.where(
        finite,
        np.clip(log_weights - maximum, -745.0, 0.0),
        -np.inf,
    )
    weights = np.where(finite, np.exp(shifted), 0.0)
    return normalize_mixture_responsibilities(weights)


def _robust_cost(
    residual: NDArray[np.float64],
    *,
    loss: MixtureLoss,
    huber_delta: float,
) -> NDArray[np.float64]:
    values = np.abs(np.asarray(residual, dtype=float))
    if loss == "squared":
        return 0.5 * values**2
    return np.where(
        values <= huber_delta,
        0.5 * values**2,
        huber_delta * (values - 0.5 * huber_delta),
    )


def _responsibility_entropy(
    responsibilities: NDArray[np.float64],
) -> float:
    positive = responsibilities > 0.0
    return float(
        -np.sum(
            responsibilities[positive]
            * np.log(responsibilities[positive])
        )
    )


def _stable_row_norms(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.hypot.reduce(np.abs(values), axis=1, initial=0.0)


def _finite_scalar(value: Any, *, field: str) -> float:
    if _contains_masked_value(value):
        raise ValueError(f"{field} must be a finite real scalar")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a finite real scalar")
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.iscomplexobj(array)
        or array.dtype.kind in "mM"
    ):
        raise ValueError(f"{field} must be a finite real scalar")
    try:
        numeric = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite real scalar") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{field} must be a finite real scalar")
    return numeric


def _as_finite_real_array(
    value: ArrayLike,
    *,
    field: str,
    ndim: int | None = None,
) -> NDArray[np.float64]:
    array = _as_real_array(value, field=field, ndim=ndim)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    return array


def _as_real_array(
    value: ArrayLike,
    *,
    field: str,
    ndim: int | None = None,
) -> NDArray[np.float64]:
    if _contains_masked_value(value):
        raise ValueError(f"{field} must not contain masked values")
    raw = np.asarray(value)
    if raw.dtype.kind in "bcmMUSV":
        raise ValueError(f"{field} must contain real numeric values")
    if raw.dtype.kind == "O":
        for item in raw.flat:
            if isinstance(
                item,
                (bool, np.bool_, complex, np.complexfloating),
            ):
                raise ValueError(
                    f"{field} must contain real numeric values"
                )
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must contain real numeric values"
        ) from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    return np.array(array, dtype=float, copy=True)


def _contains_masked_value(value: Any) -> bool:
    if value is np.ma.masked or np.ma.is_masked(value):
        return True
    if isinstance(value, np.ma.MaskedArray):
        return bool(np.any(np.ma.getmaskarray(value)))
    if isinstance(value, np.ndarray) and value.dtype == object:
        return any(_contains_masked_value(item) for item in value.flat)
    if isinstance(value, (list, tuple)):
        return any(_contains_masked_value(item) for item in value)
    return False


__all__ = [
    "GaussianMixtureMeasurementEvaluation",
    "GaussianMixtureMeasurementFactor",
    "GaussianMixtureMomentMatch",
    "MixtureLoss",
    "balance_mixture_responsibilities",
    "blend_mixture_responsibilities_with_uniform",
    "normalize_mixture_responsibilities",
]
