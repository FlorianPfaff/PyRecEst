"""Chi-square consistency diagnostics for normalized innovations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

DEFAULT_NIS_GATE_PROBABILITIES = (0.95, 0.99)
INNOVATION_COVARIANCE_SCALE_METHODS = ("mean", "quantile")


@dataclass(frozen=True)
class NISCoverageSummary:
    """Coverage and scale diagnostics at one chi-square probability."""

    probability: float
    threshold: float
    expected_fraction: float
    actual_fraction: float | None
    coverage_gap: float | None
    observed_quantile: float | None
    innovation_covariance_scale: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class NISConsistencySummary:
    """Chi-square consistency statistics for one homogeneous NIS sample."""

    measurement_dim: int
    count: int
    nis_mean: float | None
    nis_std: float | None
    nis_median: float | None
    nis_p90: float | None
    nis_p95: float | None
    nis_p99: float | None
    nis_max: float | None
    chi2_mean_expected: float
    mean_innovation_covariance_scale: float | None
    chi2_ks_distance: float | None
    coverage: tuple[NISCoverageSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        payload = asdict(self)
        payload["coverage"] = [item.to_dict() for item in self.coverage]
        return payload

    def coverage_for(self, probability: float) -> NISCoverageSummary:
        """Return the coverage entry for an exactly configured probability."""

        requested = _validate_probability(probability, "probability")
        for item in self.coverage:
            if item.probability == requested:
                return item
        raise KeyError(f"no coverage summary configured for probability {requested}")


@dataclass(frozen=True)
class InnovationCovarianceScaleEstimate:
    """Scalar innovation-covariance scale inferred from NIS statistics.

    The estimate scales the complete innovation covariance ``S``. It is not, in
    general, an exact measurement-noise covariance ``R`` multiplier because
    ``S = H P H.T + R``.
    """

    measurement_dim: int
    count: int
    method: str
    statistic: float
    target: float
    scale: float
    quantile: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def summarize_nis_consistency(
    nis_values: Iterable[float],
    measurement_dim: int,
    *,
    gate_probabilities: Iterable[float] = DEFAULT_NIS_GATE_PROBABILITIES,
) -> NISConsistencySummary:
    """Summarize agreement of NIS samples with ``chi2(measurement_dim)``.

    Samples should normally be taken before innovation gating. Accepted-only
    samples follow a truncated distribution and therefore do not provide an
    unbiased consistency assessment of the original innovation model.
    """

    dim = _positive_integer(measurement_dim, "measurement_dim")
    values = _as_nis_values(nis_values)
    probabilities = tuple(
        _validate_probability(value, "gate probability") for value in gate_probabilities
    )

    count = int(values.size)
    expected_mean = float(dim)
    if count == 0:
        coverage = tuple(
            NISCoverageSummary(
                probability=probability,
                threshold=float(chi2.ppf(probability, df=dim)),
                expected_fraction=probability,
                actual_fraction=None,
                coverage_gap=None,
                observed_quantile=None,
                innovation_covariance_scale=None,
            )
            for probability in probabilities
        )
        return NISConsistencySummary(
            measurement_dim=dim,
            count=0,
            nis_mean=None,
            nis_std=None,
            nis_median=None,
            nis_p90=None,
            nis_p95=None,
            nis_p99=None,
            nis_max=None,
            chi2_mean_expected=expected_mean,
            mean_innovation_covariance_scale=None,
            chi2_ks_distance=None,
            coverage=coverage,
        )

    nis_mean = float(np.mean(values))
    sorted_values = np.sort(values)
    empirical_cdf_upper = np.arange(1, count + 1, dtype=float) / float(count)
    empirical_cdf_lower = np.arange(count, dtype=float) / float(count)
    theoretical_cdf = chi2.cdf(sorted_values, df=dim)
    ks_distance = float(
        max(
            np.max(empirical_cdf_upper - theoretical_cdf),
            np.max(theoretical_cdf - empirical_cdf_lower),
        )
    )

    coverage_items: list[NISCoverageSummary] = []
    for probability in probabilities:
        threshold = float(chi2.ppf(probability, df=dim))
        actual_fraction = float(np.mean(values <= threshold))
        observed_quantile = float(np.quantile(values, probability))
        coverage_items.append(
            NISCoverageSummary(
                probability=probability,
                threshold=threshold,
                expected_fraction=probability,
                actual_fraction=actual_fraction,
                coverage_gap=actual_fraction - probability,
                observed_quantile=observed_quantile,
                innovation_covariance_scale=observed_quantile / threshold,
            )
        )

    return NISConsistencySummary(
        measurement_dim=dim,
        count=count,
        nis_mean=nis_mean,
        nis_std=float(np.std(values, ddof=1)) if count > 1 else 0.0,
        nis_median=float(np.percentile(values, 50.0)),
        nis_p90=float(np.percentile(values, 90.0)),
        nis_p95=float(np.percentile(values, 95.0)),
        nis_p99=float(np.percentile(values, 99.0)),
        nis_max=float(np.max(values)),
        chi2_mean_expected=expected_mean,
        mean_innovation_covariance_scale=nis_mean / expected_mean,
        chi2_ks_distance=ks_distance,
        coverage=tuple(coverage_items),
    )


def estimate_innovation_covariance_scale(
    nis_values: Iterable[float],
    measurement_dim: int,
    *,
    method: str = "mean",
    quantile: float = 0.95,
) -> InnovationCovarianceScaleEstimate:
    """Estimate a scalar multiplier for the complete innovation covariance.

    ``method='mean'`` matches the empirical mean NIS to the chi-square mean.
    ``method='quantile'`` matches an empirical quantile to the corresponding
    chi-square quantile. The caller remains responsible for clipping, shrinkage,
    minimum sample counts, and deciding whether the estimate is operationally
    suitable as a measurement-noise covariance multiplier.
    """

    dim = _positive_integer(measurement_dim, "measurement_dim")
    values = _as_nis_values(nis_values)
    if values.size == 0:
        raise ValueError("nis_values must contain at least one value")
    parsed_method = str(method).strip().lower()
    if parsed_method not in INNOVATION_COVARIANCE_SCALE_METHODS:
        raise ValueError(
            "method must be one of "
            f"{INNOVATION_COVARIANCE_SCALE_METHODS}"
        )

    if parsed_method == "mean":
        statistic = float(np.mean(values))
        target = float(dim)
        quantile_value: float | None = None
    else:
        quantile_value = _validate_probability(quantile, "quantile")
        statistic = float(np.quantile(values, quantile_value))
        target = float(chi2.ppf(quantile_value, df=dim))

    return InnovationCovarianceScaleEstimate(
        measurement_dim=dim,
        count=int(values.size),
        method=parsed_method,
        statistic=statistic,
        target=target,
        scale=statistic / target,
        quantile=quantile_value,
    )


def _as_nis_values(values: Iterable[float]) -> np.ndarray:
    message = "nis_values must contain finite non-negative real numeric values"
    if isinstance(values, (str, bytes, bytearray)) or _contains_masked_values(values):
        raise ValueError(message)
    try:
        raw_values = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if raw_values.ndim == 0:
        raw_values = raw_values.reshape(1)
    raw_values = raw_values.reshape(-1)
    if raw_values.dtype.kind in "USbcMm":
        raise ValueError(message)
    if raw_values.dtype.kind == "O":
        invalid_types = (
            bool,
            np.bool_,
            str,
            bytes,
            bytearray,
            complex,
            np.complexfloating,
            np.datetime64,
            np.timedelta64,
            type(None),
        )
        if any(isinstance(value, invalid_types) for value in raw_values):
            raise ValueError(message)
    try:
        parsed = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed).all() or np.any(parsed < 0.0):
        raise ValueError(message)
    return parsed


def _contains_masked_values(value: Any) -> bool:
    if value is np.ma.masked:
        return True
    if np.ma.isMaskedArray(value):
        try:
            if bool(np.any(np.ma.getmaskarray(value))):
                return True
        except (TypeError, ValueError):
            return True
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, np.ndarray) and value.dtype != object:
        return False
    try:
        items = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(
        item is np.ma.masked
        or (np.ma.isMaskedArray(item) and bool(np.any(np.ma.getmaskarray(item))))
        for item in items
    )


def _positive_integer(value: Any, name: str) -> int:
    message = f"{name} must be a positive integer"
    if _contains_masked_values(value) or isinstance(
        value, (bool, np.bool_, np.datetime64, np.timedelta64)
    ):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != () or array.dtype.kind in "bUSOcmM":
        raise ValueError(message)
    try:
        parsed_float = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed_float) or not parsed_float.is_integer() or parsed_float <= 0.0:
        raise ValueError(message)
    return int(parsed_float)


def _validate_probability(value: Any, name: str) -> float:
    message = f"{name} must be a finite scalar in (0, 1)"
    if _contains_masked_values(value) or isinstance(
        value,
        (bool, np.bool_, str, bytes, bytearray, np.datetime64, np.timedelta64),
    ):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != () or array.dtype.kind in "bUSOcmM":
        raise ValueError(message)
    try:
        probability = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(message)
    return probability


__all__ = [
    "DEFAULT_NIS_GATE_PROBABILITIES",
    "INNOVATION_COVARIANCE_SCALE_METHODS",
    "InnovationCovarianceScaleEstimate",
    "NISConsistencySummary",
    "NISCoverageSummary",
    "estimate_innovation_covariance_scale",
    "summarize_nis_consistency",
]
