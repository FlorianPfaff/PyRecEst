"""High-level sequence runner for SO(3)^K particle filters."""

from __future__ import annotations

from collections.abc import Callable
from operator import index as operator_index
from typing import Any

import numpy as np

# pylint: disable=no-name-in-module,no-member,too-many-positional-arguments
from pyrecest.backend import array, isfinite, ndim, ones, reshape, stack, to_numpy
from pyrecest.diagnostics import ParticleFilterResult
from pyrecest.distributions._so3_helpers import geodesic_distance
from pyrecest.distributions.so3_tangent_gaussian_distribution import (
    SO3TangentGaussianDistribution,
)

from .block_particle_filter import resolve_partition
from .so3_product_block_particle_filter import SO3ProductBlockParticleFilter
from .so3_product_particle_filter import SO3ProductParticleFilter

TransitionCallback = Callable[[Any, int, np.random.Generator], Any]


def _as_measurement_sequence(measurements):
    measurements = array(measurements, dtype=float)
    if ndim(measurements) != 3 or measurements.shape[-1] != 4:
        raise ValueError("measurements must have shape (n_steps, num_rotations, 4).")
    if measurements.shape[0] <= 0 or measurements.shape[1] <= 0:
        raise ValueError(
            "measurements must contain at least one time step and rotation."
        )
    flat = reshape(measurements, (-1, measurements.shape[1], 4))
    normalized = SO3ProductParticleFilter._as_particle_array(
        flat, measurements.shape[1]
    )
    return reshape(normalized, measurements.shape)


def _as_mask(mask, n_steps: int, num_rotations: int):
    if mask is None:
        return ones((n_steps, num_rotations))
    mask = array(mask, dtype=float)
    if mask.shape != (n_steps, num_rotations):
        raise ValueError(
            "mask must have shape " f"({n_steps}, {num_rotations}), got {mask.shape}."
        )
    if not bool(to_numpy(isfinite(mask)).all()):
        raise ValueError("mask values must be finite.")
    return array(to_numpy(mask) > 0.0, dtype=float)


def _as_confidence(confidence, mask, n_steps: int, num_rotations: int):
    if confidence is None:
        return mask
    confidence = array(confidence, dtype=float)
    if confidence.shape != (n_steps, num_rotations):
        raise ValueError(
            "confidence must have shape "
            f"({n_steps}, {num_rotations}), got {confidence.shape}."
        )
    confidence_np = to_numpy(confidence)
    if not np.all(np.isfinite(confidence_np)):
        raise ValueError("confidence values must be finite.")
    if np.any((confidence_np < 0.0) | (confidence_np > 1.0)):
        raise ValueError("confidence values must be in [0, 1].")
    return array(confidence_np * to_numpy(mask), dtype=float)


def _as_component_noise_sequence(component_noise_std, n_steps: int, num_rotations: int):
    if component_noise_std is None:
        return None
    component_noise_std = array(component_noise_std, dtype=float)
    if ndim(component_noise_std) == 0:
        values = np.full((n_steps, num_rotations), float(to_numpy(component_noise_std)))
    elif component_noise_std.shape == (num_rotations,):
        values = np.repeat(to_numpy(component_noise_std)[None, :], n_steps, axis=0)
    elif component_noise_std.shape == (n_steps, num_rotations):
        values = to_numpy(component_noise_std)
    else:
        raise ValueError(
            "component_noise_std must be scalar, have shape "
            f"({num_rotations},), or have shape ({n_steps}, {num_rotations})."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("component_noise_std values must be positive and finite.")
    return array(values, dtype=float)


def _initial_particles_from_measurement(
    first_measurement,
    first_mask,
    num_particles: int,
    initial_noise_std: float,
    rng: np.random.Generator,
):
    first_measurement_np = to_numpy(first_measurement)
    first_mask_np = to_numpy(first_mask).reshape(-1) > 0.0
    num_rotations = first_measurement_np.shape[0]
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    base = np.repeat(first_measurement_np[None, :, :], num_particles, axis=0)
    base[:, ~first_mask_np, :] = identity

    if initial_noise_std <= 0.0:
        return array(base, dtype=float)

    tangent_noise = rng.normal(
        loc=0.0,
        scale=float(initial_noise_std),
        size=(num_particles, num_rotations, 3),
    )
    components = []
    for component_idx in range(num_rotations):
        components.append(
            SO3TangentGaussianDistribution.exp_map(
                array(tangent_noise[:, component_idx, :], dtype=float),
                base=array(base[:, component_idx, :], dtype=float),
            )
        )
    return stack(components, axis=1)


def _make_filter_state(
    measurements,
    mask,
    num_particles: int,
    noise_std,
    partition,
    initial_particles,
    initial_noise_std,
    rng: np.random.Generator,
):
    num_rotations = measurements.shape[1]
    if initial_particles is None:
        if initial_noise_std is None:
            initial_noise_std = (
                0.25 * float(noise_std) if noise_std is not None else 1e-3
            )
        initial_particles = _initial_particles_from_measurement(
            measurements[0],
            mask[0],
            num_particles,
            float(initial_noise_std),
            rng,
        )

    resolved_partition = resolve_partition(num_rotations, partition)
    if len(resolved_partition) == 1:
        return (
            SO3ProductParticleFilter(
                num_particles,
                num_rotations,
                initial_particles=initial_particles,
            ),
            resolved_partition,
            False,
        )
    return (
        SO3ProductBlockParticleFilter(
            num_particles,
            num_rotations,
            partition=resolved_partition,
            initial_particles=initial_particles,
        ),
        resolved_partition,
        True,
    )


def _apply_proposal_correction(
    filter_state, measurement, confidence, proposal_gain: float
):
    if proposal_gain <= 0.0:
        return
    particles_np = to_numpy(filter_state.particles)
    measurement_np = to_numpy(measurement)
    confidence_np = to_numpy(confidence).reshape(-1)
    components = []
    for component_idx in range(filter_state.num_rotations):
        base = array(particles_np[:, component_idx, :], dtype=float)
        target = array(
            np.repeat(
                measurement_np[None, component_idx, :],
                filter_state.n_particles,
                axis=0,
            ),
            dtype=float,
        )
        delta = SO3TangentGaussianDistribution.log_map(target, base=base)
        scaled_delta = (
            float(proposal_gain) * float(confidence_np[component_idx]) * to_numpy(delta)
        )
        components.append(
            SO3TangentGaussianDistribution.exp_map(
                array(scaled_delta, dtype=float),
                base=base,
            )
        )
    filter_state.set_particles(stack(components, axis=1))


def _particle_spread(filter_state, estimate) -> float:
    distances = []
    for component_idx in range(filter_state.num_rotations):
        component_distances = to_numpy(
            geodesic_distance(
                filter_state.particles[:, component_idx, :],
                estimate[component_idx, :],
            )
        ).reshape(-1)
        if hasattr(filter_state, "component_weights"):
            weights = to_numpy(filter_state.component_weights(component_idx)).reshape(
                -1
            )
        else:
            weights = to_numpy(filter_state.weights).reshape(-1)
        distances.append(float(np.sum(weights * component_distances)))
    return float(np.mean(distances))


def _threshold_value(resample_threshold: float, num_particles: int) -> float:
    threshold = float(resample_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("resample_threshold must be nonnegative and finite.")
    if threshold == 0.0:
        return 0.0
    if threshold <= 1.0:
        return threshold * num_particles
    return threshold


def _validate_positive_integer(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        value = operator_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite.")
    return value


def _validate_positive_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return value


def _resample_if_needed(
    filter_state, ess, threshold: float, is_block_filter: bool
) -> bool:
    if threshold <= 0.0:
        return False
    if is_block_filter:
        ess_np = to_numpy(ess).reshape(-1)
        block_indices = [
            block_idx
            for block_idx, block_ess in enumerate(ess_np)
            if block_ess < threshold
        ]
        if block_indices:
            filter_state.resample_blocks_systematic(block_indices)
            return True
        return False

    if float(to_numpy(ess)) < threshold:
        filter_state.resample_systematic()
        return True
    return False


class SO3ProductSequenceFilterRunner:
    """Reusable runner for SO(3)^K particle filtering over a time series.

    The transition callback receives the current particle array, the destination
    time index, and the NumPy random generator. It must return predicted
    particles shaped ``(n_particles, num_rotations, 4)`` using scalar-last unit
    quaternions.
    """

    def __init__(
        self,
        transition_callback: TransitionCallback | None = None,
        *,
        noise_std: float | None = None,
        num_particles: int = 256,
        resample_threshold: float = 0.5,
        partition=None,
        proposal_gain: float = 0.0,
        initial_noise_std: float | None = None,
        max_noise_std: float | None = None,
        confidence_exponent: float = 1.0,
        outlier_prob: float = 0.0,
    ) -> None:
        if transition_callback is not None and not callable(transition_callback):
            raise ValueError("transition_callback must be callable or None.")
        num_particles = _validate_positive_integer("num_particles", num_particles)
        noise_std = _validate_positive_finite("noise_std", noise_std)
        max_noise_std = _validate_positive_finite("max_noise_std", max_noise_std)
        if max_noise_std is not None and noise_std is None:
            raise ValueError("noise_std is required when max_noise_std is used.")
        if max_noise_std is not None and max_noise_std < noise_std:
            raise ValueError(
                "max_noise_std must be greater than or equal to noise_std."
            )
        _threshold_value(resample_threshold, num_particles)
        proposal_gain = _validate_nonnegative_finite("proposal_gain", proposal_gain)
        if initial_noise_std is not None:
            initial_noise_std = _validate_nonnegative_finite(
                "initial_noise_std", initial_noise_std
            )
        confidence_exponent = _validate_positive_finite(
            "confidence_exponent", confidence_exponent
        )
        outlier_prob = _validate_probability("outlier_prob", outlier_prob)

        self.transition_callback = transition_callback
        self.noise_std = noise_std
        self.num_particles = num_particles
        self.resample_threshold = float(resample_threshold)
        self.partition = partition
        self.proposal_gain = proposal_gain
        self.initial_noise_std = initial_noise_std
        self.max_noise_std = max_noise_std
        self.confidence_exponent = confidence_exponent
        self.outlier_prob = outlier_prob

    def run(
        self,
        measurements,
        mask=None,
        *,
        confidence=None,
        component_noise_std=None,
        initial_particles=None,
        rng: np.random.Generator | None = None,
    ) -> ParticleFilterResult:
        """Run the configured sequence filter."""
        return run_so3_product_sequence_filter(
            measurements,
            mask,
            self.transition_callback,
            noise_std=self.noise_std,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold,
            partition=self.partition,
            proposal_gain=self.proposal_gain,
            confidence=confidence,
            component_noise_std=component_noise_std,
            initial_particles=initial_particles,
            initial_noise_std=self.initial_noise_std,
            rng=rng,
            max_noise_std=self.max_noise_std,
            confidence_exponent=self.confidence_exponent,
            outlier_prob=self.outlier_prob,
        )


def run_so3_product_sequence_filter(
    measurements,
    mask=None,
    transition_callback: TransitionCallback | None = None,
    *,
    noise_std: float | None = None,
    num_particles: int = 256,
    resample_threshold: float = 0.5,
    partition=None,
    proposal_gain: float = 0.0,
    confidence=None,
    component_noise_std=None,
    initial_particles=None,
    initial_noise_std: float | None = None,
    rng: np.random.Generator | None = None,
    max_noise_std: float | None = None,
    confidence_exponent: float = 1.0,
    outlier_prob: float = 0.0,
) -> ParticleFilterResult:
    """Filter a sequence of masked SO(3)^K measurements.

    Parameters
    ----------
    measurements
        Array shaped ``(n_steps, num_rotations, 4)`` containing scalar-last SO(3)
        quaternions.
    mask
        Optional visibility mask shaped ``(n_steps, num_rotations)``. Nonzero
        entries activate a component's measurement update.
    transition_callback
        Optional callback ``f(particles, time_index, rng)`` returning predicted
        particles for ``time_index``. With ``None``, identity dynamics are used.
    noise_std, component_noise_std
        Homoskedastic or per-component geodesic measurement noise in radians.
    partition
        ``None`` or ``"global"`` for a global product filter, ``"singleton"``
        or ``"contiguous"`` for generic block filters, or an explicit partition.
    """
    measurements = _as_measurement_sequence(measurements)
    n_steps, num_rotations = int(measurements.shape[0]), int(measurements.shape[1])
    if transition_callback is not None and not callable(transition_callback):
        raise ValueError("transition_callback must be callable or None.")
    num_particles = _validate_positive_integer("num_particles", num_particles)
    noise_std = _validate_positive_finite("noise_std", noise_std)
    max_noise_std = _validate_positive_finite("max_noise_std", max_noise_std)
    confidence_exponent = _validate_positive_finite(
        "confidence_exponent", confidence_exponent
    )
    outlier_prob = _validate_probability("outlier_prob", outlier_prob)
    if initial_noise_std is not None:
        initial_noise_std = _validate_nonnegative_finite(
            "initial_noise_std", initial_noise_std
        )
    proposal_gain = _validate_nonnegative_finite("proposal_gain", proposal_gain)
    if component_noise_std is not None and max_noise_std is not None:
        raise ValueError(
            "component_noise_std and max_noise_std are mutually exclusive."
        )
    if noise_std is None and component_noise_std is None and max_noise_std is None:
        raise ValueError(
            "noise_std, component_noise_std, or max_noise_std is required."
        )
    if max_noise_std is not None and noise_std is None:
        raise ValueError("noise_std is required when max_noise_std is used.")
    if max_noise_std is not None and max_noise_std < noise_std:
        raise ValueError("max_noise_std must be greater than or equal to noise_std.")

    rng = np.random.default_rng() if rng is None else rng
    mask = _as_mask(mask, n_steps, num_rotations)
    confidence = _as_confidence(confidence, mask, n_steps, num_rotations)
    component_noise_std = _as_component_noise_sequence(
        component_noise_std, n_steps, num_rotations
    )

    filter_state, resolved_partition, is_block_filter = _make_filter_state(
        measurements,
        mask,
        num_particles,
        noise_std,
        partition,
        initial_particles,
        initial_noise_std,
        rng,
    )

    estimates = []
    ess_history = []
    resampling_flags = []
    particle_spread = []
    block_ess_history = []
    threshold = _threshold_value(resample_threshold, num_particles)

    for time_index in range(n_steps):
        if time_index > 0 and transition_callback is not None:
            predicted = transition_callback(filter_state.particles, time_index, rng)
            filter_state.set_particles(predicted)

        _apply_proposal_correction(
            filter_state,
            measurements[time_index],
            confidence[time_index],
            proposal_gain,
        )

        update_kwargs = {
            "measurement": measurements[time_index],
            "noise_std": noise_std,
            "component_noise_std": (
                None if component_noise_std is None else component_noise_std[time_index]
            ),
            "mask": mask[time_index],
            "confidence": confidence[time_index],
            "max_noise_std": max_noise_std,
            "confidence_exponent": confidence_exponent,
            "outlier_prob": outlier_prob,
            "resample": False,
        }
        ess = filter_state.update_with_geodesic_log_likelihood(**update_kwargs)
        estimate = filter_state.mean()

        estimates.append(estimate)
        particle_spread.append(_particle_spread(filter_state, estimate))
        if is_block_filter:
            ess_per_block = filter_state.block_effective_sample_size()
            block_ess_history.append(ess_per_block)
            ess_history.append(float(np.mean(to_numpy(ess_per_block))))
            resampled = _resample_if_needed(
                filter_state, ess_per_block, threshold, is_block_filter=True
            )
        else:
            ess_history.append(float(to_numpy(ess)))
            resampled = _resample_if_needed(
                filter_state, ess, threshold, is_block_filter=False
            )
        resampling_flags.append(bool(resampled))

    metadata = {
        "num_particles": num_particles,
        "partition": resolved_partition,
        "particle_spread_unit": "rad",
    }
    return ParticleFilterResult(
        estimates=stack(estimates, axis=0),
        effective_sample_size=array(ess_history, dtype=float),
        resampled=array(resampling_flags, dtype=bool),
        particle_spread=array(particle_spread, dtype=float),
        block_effective_sample_size=(
            stack(block_ess_history, axis=0) if block_ess_history else None
        ),
        metadata=metadata,
    )


run_so3_product_particle_filter_sequence = run_so3_product_sequence_filter
