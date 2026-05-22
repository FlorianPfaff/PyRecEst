"""Point-process SCGP tracker variant for DVS event batches."""

from __future__ import annotations

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array

from .event_likelihood import (
    ContourSample,
    PointProcessUpdateConfig,
    event_batch_log_likelihood_terms,
)
from .trackers import DVSFullSCGPTracker


class DVSPointProcessSCGPTracker(DVSFullSCGPTracker):
    """Experimental SCGP tracker using a point-process event likelihood.

    The update maximizes a contour-conditioned inhomogeneous point-process
    likelihood. It is intentionally kept separate from ``DVSFullSCGPTracker`` so
    that the current activity-weighted update remains available as a baseline.
    """

    def __init__(
        self,
        *args,
        point_process_update_config=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.point_process_update_config = (
            point_process_update_config or PointProcessUpdateConfig()
        )
        self.last_event_likelihood_terms = None
        self.last_event_likelihood_gradient = None
        self.last_event_likelihood_state_update = None
        self.last_event_log_likelihood = None
        self.last_event_activities = array([])
        self.last_active_measurement_indices = []
        self.last_quadratic_form = None
        self.last_event_signed_normal_flows = None
        self.last_event_polarity_consistencies = None
        self.last_event_polarity_weights = None
        self.last_polarity_contrast_sign = None

    def sample_contour(self, n=100):
        """Return sampled star-convex contour geometry for likelihood models."""
        if n <= 2:
            raise ValueError("n must be greater than 2")

        orientation = float(self.kinematic_state[2])
        position = np.asarray(self.kinematic_state[:2], dtype=float)
        shape_state = np.asarray(self.shape_state, dtype=float)
        body_angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        delta_angle = 2.0 * np.pi / float(n)
        points = []
        normals = []
        weights = []
        for body_angle in body_angles:
            world_angle = float(body_angle + orientation)
            unit_direction = np.array(
                [np.cos(world_angle), np.sin(world_angle)], dtype=float
            )
            body_angle_vector = array([float(body_angle)])
            basis_row = np.asarray(
                self._basis_matrix(body_angle_vector)[0], dtype=float
            )
            derivative_row = np.asarray(
                self._basis_derivative(body_angle_vector)[0], dtype=float
            )
            radius = float(basis_row @ shape_state)
            radius_derivative = float(derivative_row @ shape_state)
            tangent = radius_derivative * unit_direction + radius * np.array(
                [-unit_direction[1], unit_direction[0]],
                dtype=float,
            )
            normal = np.array([tangent[1], -tangent[0]], dtype=float)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-12:
                normal = unit_direction
            else:
                normal = normal / normal_norm
            points.append(position + radius * unit_direction)
            normals.append(normal)
            weights.append(float(np.linalg.norm(tangent)) * delta_angle)

        return ContourSample(
            points=np.asarray(points, dtype=float),
            normals=np.asarray(normals, dtype=float),
            weights=np.asarray(weights, dtype=float),
            angles=body_angles,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def update(
        self,
        measurements,
        R=None,
        s_hat=None,
        sigma_squared_s=None,
        event_velocity=None,
        point_process_update_config=None,
        batch_duration=None,
        event_polarities=None,
        event_signed_normal_flows=None,
        polarity_mismatch_weight=None,
        polarity_contrast_sign=None,
        image_area=None,
    ):
        """Update the tracker state with the point-process event likelihood."""
        del R
        if s_hat is not None:
            self.scale_mean = float(s_hat)
        if sigma_squared_s is not None:
            self.scale_variance = float(sigma_squared_s)
        return self.update_event_batch(
            measurements,
            event_velocity=event_velocity,
            point_process_update_config=point_process_update_config,
            batch_duration=batch_duration,
            event_polarities=event_polarities,
            event_signed_normal_flows=event_signed_normal_flows,
            polarity_mismatch_weight=polarity_mismatch_weight,
            polarity_contrast_sign=polarity_contrast_sign,
            image_area=image_area,
        )

    def update_event_batch(
        self,
        measurements,
        *,
        event_velocity=None,
        point_process_update_config=None,
        batch_duration=None,
        event_polarities=None,
        event_signed_normal_flows=None,
        polarity_mismatch_weight=None,
        polarity_contrast_sign=None,
        image_area=None,
    ):
        """Run a MAP-style point-process likelihood update for one event batch."""
        config = point_process_update_config or self.point_process_update_config
        measurements = np.asarray(
            self._normalize_measurements(measurements), dtype=float
        )
        velocity = np.asarray(self._get_event_velocity(event_velocity), dtype=float)
        measurement_count = measurements.shape[0]
        event_polarities = self._normalize_event_polarities(
            event_polarities, measurement_count
        )
        event_signed_normal_flows = self._normalize_event_signed_normal_flows(
            event_signed_normal_flows, measurement_count
        )
        if polarity_mismatch_weight is None:
            polarity_mismatch_weight = self.polarity_mismatch_weight
        else:
            polarity_mismatch_weight = self._validate_polarity_mismatch_weight(
                polarity_mismatch_weight
            )
        (
            signed_flows,
            activities,
            resolved_polarity_contrast_sign,
            polarity_consistencies,
            polarity_weights,
            _measurement_weights,
            active_measurement_mask,
        ) = self._event_measurement_weights(
            measurements,
            velocity,
            event_polarities,
            self.event_activity_floor,
            self.inactive_activity_threshold,
            polarity_mismatch_weight,
            polarity_contrast_sign,
            event_signed_normal_flows=event_signed_normal_flows,
        )
        active_measurement_indices = [
            index
            for index, is_active in enumerate(active_measurement_mask)
            if bool(is_active)
        ]
        active_index_array = np.asarray(active_measurement_indices, dtype=np.int64)
        update_measurements = measurements[active_index_array]

        if update_measurements.shape[0] == 0:
            terms = self._likelihood_terms_for_current_state(
                update_measurements, velocity, config, batch_duration, image_area
            )
            zero_update = np.zeros_like(self._state_as_numpy())
            self._record_event_likelihood_diagnostics(
                measurements=measurements,
                velocity=velocity,
                config=config,
                gradient=zero_update,
                state_update=zero_update,
                terms=terms,
                signed_flows=signed_flows,
                activities=activities,
                active_measurement_indices=active_measurement_indices,
                polarity_consistencies=(
                    None
                    if event_polarities is None
                    or resolved_polarity_contrast_sign is None
                    else polarity_consistencies
                ),
                polarity_weights=(
                    None
                    if event_polarities is None
                    or resolved_polarity_contrast_sign is None
                    else polarity_weights
                ),
                resolved_polarity_contrast_sign=resolved_polarity_contrast_sign,
            )
            if self.log_posterior_estimates:
                self.store_posterior_estimates()
            if self.log_posterior_extents:
                self.store_posterior_extents()
            return

        gradient = np.zeros_like(self._state_as_numpy())
        state_update = np.zeros_like(gradient)
        for _ in range(config.max_map_iterations):
            state = self._state_as_numpy()
            gradient = self._finite_difference_log_likelihood_gradient(
                state,
                update_measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            covariance = np.asarray(self.covariance, dtype=float)
            state_update = config.map_step_size * (covariance @ gradient)
            state_update = self._clip_state_update(
                state_update, config.max_state_update_norm
            )
            if float(np.linalg.norm(state_update)) <= 1e-12:
                break
            self._set_state_from_numpy(state + state_update)
            self.covariance = self._symmetrize(
                array(config.covariance_damping * np.asarray(self.covariance))
            )

        terms = self._likelihood_terms_for_current_state(
            update_measurements, velocity, config, batch_duration, image_area
        )
        self._record_event_likelihood_diagnostics(
            measurements=measurements,
            velocity=velocity,
            config=config,
            gradient=gradient,
            state_update=state_update,
            terms=terms,
            signed_flows=signed_flows,
            activities=activities,
            active_measurement_indices=active_measurement_indices,
            polarity_consistencies=(
                None
                if event_polarities is None or resolved_polarity_contrast_sign is None
                else polarity_consistencies
            ),
            polarity_weights=(
                None
                if event_polarities is None or resolved_polarity_contrast_sign is None
                else polarity_weights
            ),
            resolved_polarity_contrast_sign=resolved_polarity_contrast_sign,
        )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _record_event_likelihood_diagnostics(
        self,
        *,
        measurements,
        velocity,
        config,
        gradient,
        state_update,
        terms,
        signed_flows=None,
        activities=None,
        active_measurement_indices=None,
        polarity_consistencies=None,
        polarity_weights=None,
        resolved_polarity_contrast_sign=None,
    ):
        """Store diagnostics for the latest point-process event update."""
        self.last_event_likelihood_terms = terms
        self.last_event_likelihood_gradient = array(gradient)
        self.last_event_likelihood_state_update = array(state_update)
        if activities is None:
            activities = self.contour_event_activity(
                n=config.contour_samples, event_velocity=velocity
            )
        self.last_event_activities = array(activities)
        if active_measurement_indices is None:
            active_measurement_indices = range(measurements.shape[0])
        self.last_active_measurement_indices = [
            int(index) for index in active_measurement_indices
        ]
        self.last_event_signed_normal_flows = (
            None if signed_flows is None else array(signed_flows)
        )
        self.last_event_log_likelihood = terms.log_likelihood
        self.last_quadratic_form = None
        self.last_event_polarity_consistencies = polarity_consistencies
        self.last_event_polarity_weights = (
            None if polarity_weights is None else array(polarity_weights)
        )
        self.last_polarity_contrast_sign = resolved_polarity_contrast_sign

    def _finite_difference_log_likelihood_gradient(
        self,
        state,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        gradient = np.zeros_like(state)
        eps = config.finite_difference_eps
        for state_index in self._likelihood_state_indices(config):
            perturbation = np.zeros_like(state)
            perturbation[state_index] = eps
            plus = self._log_likelihood_for_state(
                state + perturbation,
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            minus = self._log_likelihood_for_state(
                state - perturbation,
                measurements,
                velocity,
                config,
                batch_duration,
                image_area,
            )
            gradient[state_index] = (plus - minus) / (2.0 * eps)
        return gradient

    def _log_likelihood_for_state(
        self,
        state,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        terms = self._likelihood_terms_for_state(
            state, measurements, velocity, config, batch_duration, image_area
        )
        return terms.log_likelihood

    def _likelihood_terms_for_current_state(
        self,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        contour = self.sample_contour(config.contour_samples)
        return event_batch_log_likelihood_terms(
            measurements,
            contour,
            velocity,
            config.likelihood,
            batch_duration=batch_duration,
            image_area=image_area,
        )

    def _likelihood_terms_for_state(
        self,
        state,
        measurements,
        velocity,
        config,
        batch_duration,
        image_area,
    ):
        original_state = self._state_as_numpy()
        try:
            self._set_state_from_numpy(state)
            return self._likelihood_terms_for_current_state(
                measurements, velocity, config, batch_duration, image_area
            )
        finally:
            self._set_state_from_numpy(original_state)

    def _likelihood_state_indices(self, config):
        state_size = self._state_as_numpy().shape[0]
        kinematic_size = np.asarray(self.kinematic_state).shape[0]
        indices = [index for index in range(min(3, kinematic_size, state_size))]
        shape_start = kinematic_size
        available_shape_count = max(0, state_size - shape_start)
        shape_count = min(
            int(config.shape_update_modes),
            np.asarray(self.shape_state).shape[0],
            available_shape_count,
        )
        indices.extend(range(shape_start, shape_start + shape_count))
        return indices

    def _state_as_numpy(self):
        return np.asarray(self.state, dtype=float).copy()

    def _set_state_from_numpy(self, state):
        self.state = array(np.asarray(state, dtype=float))
        self._sync_state_views()

    @staticmethod
    def _clip_state_update(state_update, max_norm):
        update_norm = float(np.linalg.norm(state_update))
        if update_norm <= max_norm:
            return state_update
        return state_update * (float(max_norm) / update_norm)


DVSPointProcessSCGP = DVSPointProcessSCGPTracker
