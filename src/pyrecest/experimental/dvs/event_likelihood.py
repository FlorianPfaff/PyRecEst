"""Point-process event likelihoods for DVS contour observations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _validate_positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class ContourSample:
    """Sampled contour geometry used by event-generation likelihoods."""

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    angles: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        angles = None
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        if normals.shape != points.shape:
            raise ValueError("normals must have the same shape as points")
        if weights.shape != (points.shape[0],):
            raise ValueError("weights must have shape (n,)")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative")
        if self.angles is not None:
            angles = np.asarray(self.angles, dtype=float)
            if angles.shape != (points.shape[0],):
                raise ValueError("angles must have shape (n,)")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "angles", angles)


@dataclass(frozen=True)
class EventLikelihoodConfig:
    """Parameters for the contour-conditioned point-process likelihood."""

    spatial_sigma_px: float = 1.5
    foreground_rate: float = 1.0
    background_rate: float = 1e-4
    activity_floor: float = 0.0
    min_intensity: float = 1e-12
    include_expected_count: bool = True
    normalize_kernel: bool = True
    batch_duration: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_finite(self.spatial_sigma_px, "spatial_sigma_px")
        _validate_nonnegative_finite(self.foreground_rate, "foreground_rate")
        _validate_nonnegative_finite(self.background_rate, "background_rate")
        _validate_nonnegative_finite(self.activity_floor, "activity_floor")
        _validate_positive_finite(self.min_intensity, "min_intensity")
        _validate_nonnegative_finite(self.batch_duration, "batch_duration")


@dataclass(frozen=True)
class PointProcessUpdateConfig:
    """MAP-style update parameters for the experimental point-process tracker."""

    likelihood: EventLikelihoodConfig = field(default_factory=EventLikelihoodConfig)
    contour_samples: int = 96
    finite_difference_eps: float = 1e-2
    map_step_size: float = 0.2
    max_map_iterations: int = 2
    shape_update_modes: int = 8
    covariance_damping: float = 0.98
    max_state_update_norm: float = 5.0

    def __post_init__(self) -> None:
        if self.contour_samples <= 2:
            raise ValueError("contour_samples must be greater than 2")
        _validate_positive_finite(self.finite_difference_eps, "finite_difference_eps")
        _validate_nonnegative_finite(self.map_step_size, "map_step_size")
        if self.max_map_iterations < 0:
            raise ValueError("max_map_iterations must be non-negative")
        if self.shape_update_modes < 0:
            raise ValueError("shape_update_modes must be non-negative")
        covariance_damping = float(self.covariance_damping)
        if not np.isfinite(covariance_damping) or not 0.0 < covariance_damping <= 1.0:
            raise ValueError("covariance_damping must be finite and in (0, 1]")
        _validate_positive_finite(self.max_state_update_norm, "max_state_update_norm")


@dataclass(frozen=True)
class EventLikelihoodTerms:
    """Diagnostic decomposition of a point-process event log likelihood."""

    log_intensity_sum: float
    expected_foreground_count: float
    expected_background_count: float
    log_likelihood: float
    mean_activity: float
    event_count: int


def normal_flow_activities(
    normals: np.ndarray,
    velocity: np.ndarray,
    activity_floor: float = 0.0,
) -> np.ndarray:
    """Return normalized normal-flow activity for sampled contour normals."""
    normals = np.asarray(normals, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 2:
        raise ValueError("normals must have shape (n, 2)")
    if velocity.shape != (2,):
        raise ValueError("velocity must have shape (2,)")
    if activity_floor < 0.0:
        raise ValueError("activity_floor must be non-negative")

    velocity_norm = float(np.linalg.norm(velocity))
    if velocity_norm <= 1e-12:
        activities = np.zeros(normals.shape[0], dtype=float)
    else:
        normal_norms = np.linalg.norm(normals, axis=1)
        unit_normals = np.divide(
            normals,
            normal_norms[:, None],
            out=np.zeros_like(normals),
            where=normal_norms[:, None] > 1e-12,
        )
        activities = np.abs(unit_normals @ velocity) / velocity_norm
    if activity_floor > 0.0:
        activities = np.maximum(activities, activity_floor)
    return activities


def contour_event_intensity(
    event_xy: np.ndarray,
    contour: ContourSample,
    velocity: np.ndarray,
    config: EventLikelihoodConfig | None = None,
) -> np.ndarray:
    """Evaluate event intensities at event coordinates for one contour state."""
    config = config or EventLikelihoodConfig()
    events = _as_event_xy(event_xy)
    activities = normal_flow_activities(
        np.asarray(contour.normals, dtype=float),
        velocity,
        activity_floor=config.activity_floor,
    )
    weights = np.asarray(contour.weights, dtype=float)
    weighted_activity = weights * activities
    if events.shape[0] == 0:
        return np.empty(0, dtype=float)

    kernel = _gaussian_contour_kernel(
        events,
        np.asarray(contour.points, dtype=float),
        sigma_px=config.spatial_sigma_px,
        normalize=config.normalize_kernel,
    )
    foreground = config.foreground_rate * (kernel @ weighted_activity)
    return config.background_rate + foreground


def expected_event_count(
    contour: ContourSample,
    velocity: np.ndarray,
    config: EventLikelihoodConfig | None = None,
    *,
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> tuple[float, float]:
    """Return expected foreground and background event counts."""
    config = config or EventLikelihoodConfig()
    duration = _duration_from_argument(config, batch_duration)
    if duration == 0.0:
        return 0.0, 0.0

    activities = normal_flow_activities(
        np.asarray(contour.normals, dtype=float),
        velocity,
        activity_floor=config.activity_floor,
    )
    expected_foreground = (
        duration
        * config.foreground_rate
        * float(np.sum(np.asarray(contour.weights, dtype=float) * activities))
    )
    expected_background = 0.0
    if image_area is not None:
        if image_area < 0.0:
            raise ValueError("image_area must be non-negative")
        expected_background = duration * config.background_rate * float(image_area)
    return expected_foreground, expected_background


def event_batch_log_likelihood_terms(
    event_xy: np.ndarray,
    contour: ContourSample,
    velocity: np.ndarray,
    config: EventLikelihoodConfig | None = None,
    *,
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> EventLikelihoodTerms:
    """Return a diagnostic point-process log-likelihood decomposition."""
    config = config or EventLikelihoodConfig()
    events = _as_event_xy(event_xy)
    intensities = contour_event_intensity(events, contour, velocity, config)
    log_intensity_sum = float(
        np.sum(np.log(np.maximum(intensities, config.min_intensity)))
    )
    expected_foreground, expected_background = expected_event_count(
        contour,
        velocity,
        config,
        batch_duration=batch_duration,
        image_area=image_area,
    )
    expected_total = (
        expected_foreground + expected_background
        if config.include_expected_count
        else 0.0
    )
    activities = normal_flow_activities(
        np.asarray(contour.normals, dtype=float),
        velocity,
        activity_floor=config.activity_floor,
    )
    return EventLikelihoodTerms(
        log_intensity_sum=log_intensity_sum,
        expected_foreground_count=float(expected_foreground),
        expected_background_count=float(expected_background),
        log_likelihood=float(log_intensity_sum - expected_total),
        mean_activity=float(np.mean(activities)) if activities.size else 0.0,
        event_count=int(events.shape[0]),
    )


def event_batch_log_likelihood(
    event_xy: np.ndarray,
    contour: ContourSample,
    velocity: np.ndarray,
    config: EventLikelihoodConfig | None = None,
    *,
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> float:
    """Return the contour-conditioned point-process event log likelihood."""
    return event_batch_log_likelihood_terms(
        event_xy,
        contour,
        velocity,
        config,
        batch_duration=batch_duration,
        image_area=image_area,
    ).log_likelihood


def scgp_event_batch_log_likelihood_terms(
    tracker,
    event_xy: np.ndarray,
    velocity: np.ndarray,
    config: EventLikelihoodConfig | PointProcessUpdateConfig | None = None,
    *,
    contour_samples: int | None = None,
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> EventLikelihoodTerms:
    """Score an event batch against a tracker-provided contour sample.

    The tracker only has to provide ``sample_contour(n=...)`` returning an
    object with ``points``, ``normals``, and ``weights`` attributes. This keeps
    the point-process likelihood reusable for SCGP trackers without requiring a
    tracker-specific likelihood implementation. ``config`` may be either an
    ``EventLikelihoodConfig`` or a ``PointProcessUpdateConfig``; in the latter
    case the embedded likelihood config and contour-sample count are used.
    """
    likelihood_config, sample_count = _resolve_scgp_likelihood_arguments(
        config,
        contour_samples,
    )
    contour = tracker.sample_contour(n=sample_count)
    return event_batch_log_likelihood_terms(
        event_xy,
        contour,
        velocity,
        likelihood_config,
        batch_duration=batch_duration,
        image_area=image_area,
    )


def scgp_event_batch_log_likelihood(
    tracker,
    event_xy: np.ndarray,
    velocity: np.ndarray,
    config: EventLikelihoodConfig | PointProcessUpdateConfig | None = None,
    *,
    contour_samples: int | None = None,
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> float:
    """Return the point-process event log likelihood for an SCGP tracker."""
    return scgp_event_batch_log_likelihood_terms(
        tracker,
        event_xy,
        velocity,
        config,
        contour_samples=contour_samples,
        batch_duration=batch_duration,
        image_area=image_area,
    ).log_likelihood


def _resolve_scgp_likelihood_arguments(
    config: EventLikelihoodConfig | PointProcessUpdateConfig | None,
    contour_samples: int | None,
) -> tuple[EventLikelihoodConfig, int]:
    if isinstance(config, PointProcessUpdateConfig):
        likelihood_config = config.likelihood
        sample_count = (
            config.contour_samples if contour_samples is None else contour_samples
        )
    else:
        likelihood_config = config or EventLikelihoodConfig()
        sample_count = 96 if contour_samples is None else contour_samples
    if int(sample_count) <= 2:
        raise ValueError("contour_samples must be greater than 2")
    return likelihood_config, int(sample_count)


def _gaussian_contour_kernel(
    event_xy: np.ndarray,
    contour_points: np.ndarray,
    *,
    sigma_px: float,
    normalize: bool,
) -> np.ndarray:
    diff = event_xy[:, None, :] - np.asarray(contour_points, dtype=float)[None, :, :]
    squared_distances = np.sum(diff * diff, axis=2)
    kernel = np.exp(-0.5 * squared_distances / (sigma_px * sigma_px))
    if normalize:
        kernel = kernel / (2.0 * np.pi * sigma_px * sigma_px)
    return kernel


def _as_event_xy(event_xy: np.ndarray) -> np.ndarray:
    events = np.asarray(event_xy, dtype=float)
    if events.ndim == 1 and events.size == 0:
        return np.empty((0, 2), dtype=float)
    if events.ndim != 2 or events.shape[1] != 2:
        raise ValueError("event_xy must have shape (n, 2)")
    return events


def _duration_from_argument(
    config: EventLikelihoodConfig,
    batch_duration: float | None,
) -> float:
    if batch_duration is None:
        duration = config.batch_duration
    else:
        duration = float(batch_duration)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("batch_duration must be finite and non-negative")
    return duration
