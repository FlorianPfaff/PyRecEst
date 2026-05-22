"""PyRecEst-based DVS-ENACT tracker variants."""

from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,too-many-instance-attributes,too-many-locals
from pyrecest.backend import (
    all,
    arctan2,
    array,
    cos,
    isfinite,
    linalg,
    linspace,
    pi,
    sin,
)
from pyrecest.filters.gprhm_tracker import FullSCGPTracker


class DVSFullSCGPTracker(FullSCGPTracker):
    """Star-convex GP tracker with a DVS-inspired active-contour update.

    Event cameras mostly observe moving brightness contours. A contour point is
    therefore most informative when the apparent image velocity has a component
    along the local contour normal. Measurements on nearly inactive contour
    parts can be skipped or strongly down-weighted instead of being interpreted
    as uniformly sampled extent returns.

    Polarity, when supplied, is treated as a sign check on the signed normal
    flow. The unknown object/background contrast sign can be fixed or inferred
    per event batch.
    """

    _POLARITY_INFER_SENTINEL = "infer"

    def __init__(
        self,
        *args,
        event_activity_floor=1e-3,
        inactive_activity_threshold=0.0,
        polarity_mismatch_weight=0.25,
        polarity_contrast_sign=_POLARITY_INFER_SENTINEL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_activity_floor = float(event_activity_floor)
        self.inactive_activity_threshold = float(inactive_activity_threshold)
        if self.event_activity_floor < 0.0:
            raise ValueError("event_activity_floor must be non-negative")
        if self.inactive_activity_threshold < 0.0:
            raise ValueError("inactive_activity_threshold must be non-negative")
        self.polarity_mismatch_weight = self._validate_polarity_mismatch_weight(
            polarity_mismatch_weight
        )
        self.polarity_contrast_sign = self._normalize_polarity_contrast_sign(
            polarity_contrast_sign
        )

        self.last_event_activities = None
        self.last_event_signed_normal_flows = None
        self.last_active_measurement_indices = None
        self.last_event_polarity_consistencies = None
        self.last_event_polarity_weights = None
        self.last_polarity_contrast_sign = None

    @staticmethod
    def _validate_polarity_mismatch_weight(value):
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("polarity_mismatch_weight must be in [0, 1]")
        return value

    @classmethod
    def _normalize_polarity_contrast_sign(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.lower()
            if normalized != cls._POLARITY_INFER_SENTINEL:
                raise ValueError(
                    "polarity_contrast_sign must be 'infer', None, or non-zero"
                )
            return cls._POLARITY_INFER_SENTINEL
        value = float(value)
        if value == 0.0:
            raise ValueError(
                "polarity_contrast_sign must be 'infer', None, or non-zero"
            )
        return 1.0 if value > 0.0 else -1.0

    @staticmethod
    def _event_polarity_sign(event_polarity):
        # MEVDT and the synthetic generator use 0/1; some datasets use -1/+1.
        return 1.0 if float(event_polarity) > 0.0 else -1.0

    @staticmethod
    def _signed_scalar_sign(value, zero_tolerance=1e-12):
        value = float(value)
        if value > zero_tolerance:
            return 1.0
        if value < -zero_tolerance:
            return -1.0
        return 0.0

    @staticmethod
    def _normalize_event_polarities(event_polarities, measurement_count):
        if event_polarities is None:
            return None
        event_polarities = array(event_polarities)
        if event_polarities.shape != (measurement_count,):
            raise ValueError("event_polarities must have shape (n_measurements,)")
        return event_polarities

    @staticmethod
    def _normalize_event_signed_normal_flows(
        event_signed_normal_flows, measurement_count
    ):
        if event_signed_normal_flows is None:
            return None
        signed_flows = array(event_signed_normal_flows)
        if signed_flows.shape != (measurement_count,):
            raise ValueError(
                "event_signed_normal_flows must have shape (n_measurements,)"
            )
        return signed_flows

    def _get_event_velocity(self, event_velocity):
        if event_velocity is None:
            if not self.velocities:
                raise ValueError(
                    "event_velocity must be provided when velocities=False"
                )
            return self.kinematic_state[3] * array(
                [cos(self.kinematic_state[2]), sin(self.kinematic_state[2])]
            )

        event_velocity = array(event_velocity)
        if event_velocity.shape != (self.measurement_dim,):
            raise ValueError("event_velocity must have shape (2,)")
        if not bool(all(isfinite(event_velocity))):
            raise ValueError("event_velocity must be finite")
        return event_velocity

    def _unit_direction_from_measurement(self, measurement):
        position = self.kinematic_state[:2]
        delta = measurement - position
        delta_norm = linalg.norm(delta)
        if float(delta_norm) <= 1e-12:
            return array([cos(self.kinematic_state[2]), sin(self.kinematic_state[2])])
        return delta / delta_norm

    def _contour_normal_from_unit_direction(self, unit_direction):
        orientation = self.kinematic_state[2]
        world_angle = arctan2(unit_direction[1], unit_direction[0])
        body_angle = world_angle - orientation
        basis_row = self._basis_matrix(body_angle)[0]
        basis_derivative_row = self._basis_derivative(body_angle)[0]
        radius = basis_row @ self.shape_state
        radius_derivative = basis_derivative_row @ self.shape_state

        tangent = radius_derivative * unit_direction + radius * array(
            [-unit_direction[1], unit_direction[0]]
        )
        normal = array([tangent[1], -tangent[0]])
        normal_norm = linalg.norm(normal)
        if float(normal_norm) <= 1e-12:
            return unit_direction
        return normal / normal_norm

    def signed_normal_flow_for_measurement(self, measurement, event_velocity=None):
        """Return signed normalized normal flow for one event measurement."""
        measurement = array(measurement)
        if measurement.shape != (self.measurement_dim,):
            raise ValueError("measurement must have shape (2,)")

        velocity = self._get_event_velocity(event_velocity)
        velocity_norm = linalg.norm(velocity)
        if float(velocity_norm) <= 1e-12:
            return 0.0

        unit_direction = self._unit_direction_from_measurement(measurement)
        normal = self._contour_normal_from_unit_direction(unit_direction)
        return float((normal @ velocity) / velocity_norm)

    def event_activity_for_measurement(self, measurement, event_velocity=None):
        """Return normalized normal-flow activity for one event measurement."""
        return abs(self.signed_normal_flow_for_measurement(measurement, event_velocity))

    def polarity_consistency_for_signed_flow(
        self,
        signed_normal_flow,
        event_polarity,
        polarity_contrast_sign=1.0,
    ):
        """Return whether an event polarity agrees with signed normal flow.

        ``polarity_contrast_sign`` captures the unknown object/background
        contrast sign. Use ``+1`` when ON events should agree with positive
        signed normal flow, ``-1`` when they should disagree. Batch-level
        inference resolves the ``"infer"`` sentinel before this method is used.
        """
        contrast_sign = self._normalize_polarity_contrast_sign(polarity_contrast_sign)
        if contrast_sign is None:
            return None
        if contrast_sign == self._POLARITY_INFER_SENTINEL:
            raise ValueError(
                "polarity_contrast_sign='infer' must be resolved at batch level"
            )

        flow_sign = self._signed_scalar_sign(signed_normal_flow)
        if flow_sign == 0.0:
            return None
        expected_sign = contrast_sign * flow_sign
        observed_sign = self._event_polarity_sign(event_polarity)
        return bool(observed_sign == expected_sign)

    def polarity_weight_for_signed_flow(
        self,
        signed_normal_flow,
        event_polarity,
        polarity_contrast_sign=1.0,
        polarity_mismatch_weight=None,
    ):
        """Return a multiplicative activity weight from polarity consistency."""
        if polarity_mismatch_weight is None:
            polarity_mismatch_weight = self.polarity_mismatch_weight
        else:
            polarity_mismatch_weight = self._validate_polarity_mismatch_weight(
                polarity_mismatch_weight
            )
        consistency = self.polarity_consistency_for_signed_flow(
            signed_normal_flow,
            event_polarity,
            polarity_contrast_sign=polarity_contrast_sign,
        )
        if consistency is None or consistency:
            return 1.0
        return polarity_mismatch_weight

    def _resolve_polarity_contrast_sign(
        self,
        signed_flows,
        event_polarities,
        polarity_contrast_sign,
    ):
        if event_polarities is None:
            return None
        if polarity_contrast_sign is None:
            polarity_contrast_sign = self.polarity_contrast_sign
        contrast_sign = self._normalize_polarity_contrast_sign(polarity_contrast_sign)
        if contrast_sign is None:
            return None
        if contrast_sign != self._POLARITY_INFER_SENTINEL:
            return contrast_sign

        polarity_flow_score = 0.0
        for signed_flow, event_polarity in zip(
            signed_flows,
            event_polarities,
            strict=True,
        ):
            if self._signed_scalar_sign(signed_flow) == 0.0:
                continue
            polarity_flow_score += self._event_polarity_sign(event_polarity) * float(
                signed_flow
            )
        if abs(polarity_flow_score) <= 1e-12:
            return 1.0
        return 1.0 if polarity_flow_score > 0.0 else -1.0

    def contour_signed_normal_flow(
        self,
        n=100,
        angles=None,
        event_velocity=None,
        body_frame=False,
    ):
        """Evaluate signed normal flow over image-plane or body-frame angles."""
        if angles is None:
            angles = linspace(0.0, 2 * pi, n, endpoint=False)
        else:
            angles = array(angles)

        velocity = self._get_event_velocity(event_velocity)
        velocity_norm = linalg.norm(velocity)
        signed_flows = []
        for angle in angles:
            world_angle = angle + self.kinematic_state[2] if body_frame else angle
            unit_direction = array([cos(world_angle), sin(world_angle)])
            normal = self._contour_normal_from_unit_direction(unit_direction)
            signed_flow = (
                0.0
                if float(velocity_norm) <= 1e-12
                else float((normal @ velocity) / velocity_norm)
            )
            signed_flows.append(signed_flow)
        return array(signed_flows)

    def contour_event_activity(
        self,
        n=100,
        angles=None,
        event_velocity=None,
        body_frame=False,
        apply_floor=False,
    ):
        """Evaluate DVS contour activity over image-plane or body-frame angles."""
        signed_flows = self.contour_signed_normal_flow(
            n=n,
            angles=angles,
            event_velocity=event_velocity,
            body_frame=body_frame,
        )
        activities = []
        for signed_flow in signed_flows:
            activity = abs(float(signed_flow))
            if apply_floor and activity < self.event_activity_floor:
                activity = self.event_activity_floor
            activities.append(activity)
        return array(activities)

    def _event_measurement_weights(
        self,
        measurements,
        velocity,
        event_polarities,
        event_activity_floor,
        inactive_activity_threshold,
        polarity_mismatch_weight,
        polarity_contrast_sign,
        event_signed_normal_flows=None,
    ):
        """Return DVS reliability weights for PyRecEst's weighted SCGP update."""
        if event_signed_normal_flows is None:
            signed_flows = [
                self.signed_normal_flow_for_measurement(measurement, velocity)
                for measurement in measurements
            ]
        else:
            signed_flows = [float(value) for value in event_signed_normal_flows]
        activities = [abs(signed_flow) for signed_flow in signed_flows]
        resolved_polarity_contrast_sign = self._resolve_polarity_contrast_sign(
            signed_flows,
            event_polarities,
            polarity_contrast_sign,
        )

        polarity_consistencies = []
        polarity_weights = []
        measurement_weights = []
        active_measurement_mask = []
        for measurement_index, activity in enumerate(activities):
            polarity_consistency = None
            polarity_weight = 1.0
            if (
                event_polarities is not None
                and resolved_polarity_contrast_sign is not None
            ):
                polarity_consistency = self.polarity_consistency_for_signed_flow(
                    signed_flows[measurement_index],
                    event_polarities[measurement_index],
                    polarity_contrast_sign=resolved_polarity_contrast_sign,
                )
                polarity_weight = self.polarity_weight_for_signed_flow(
                    signed_flows[measurement_index],
                    event_polarities[measurement_index],
                    polarity_contrast_sign=resolved_polarity_contrast_sign,
                    polarity_mismatch_weight=polarity_mismatch_weight,
                )
            polarity_consistencies.append(polarity_consistency)
            polarity_weights.append(polarity_weight)

            weighted_activity = activity * polarity_weight
            is_active = (
                weighted_activity > 0.0
                and weighted_activity >= inactive_activity_threshold
            )
            active_measurement_mask.append(is_active)
            measurement_weights.append(
                max(weighted_activity, event_activity_floor) if is_active else 0.0
            )

        return (
            signed_flows,
            activities,
            resolved_polarity_contrast_sign,
            polarity_consistencies,
            polarity_weights,
            array(measurement_weights),
            active_measurement_mask,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def update(
        self,
        measurements,
        R=None,
        s_hat=None,
        sigma_squared_s=None,
        event_velocity=None,
        event_activity_floor=None,
        inactive_activity_threshold=None,
        event_polarities=None,
        event_signed_normal_flows=None,
        polarity_mismatch_weight=None,
        polarity_contrast_sign=None,
    ):
        if s_hat is not None:
            self.scale_mean = float(s_hat)
        if sigma_squared_s is not None:
            self.scale_variance = float(sigma_squared_s)
        if event_activity_floor is None:
            event_activity_floor = self.event_activity_floor
        else:
            event_activity_floor = float(event_activity_floor)
        if inactive_activity_threshold is None:
            inactive_activity_threshold = self.inactive_activity_threshold
        else:
            inactive_activity_threshold = float(inactive_activity_threshold)
        if event_activity_floor < 0.0:
            raise ValueError("event_activity_floor must be non-negative")
        if inactive_activity_threshold < 0.0:
            raise ValueError("inactive_activity_threshold must be non-negative")
        if polarity_mismatch_weight is None:
            polarity_mismatch_weight = self.polarity_mismatch_weight
        else:
            polarity_mismatch_weight = self._validate_polarity_mismatch_weight(
                polarity_mismatch_weight
            )

        measurements = self._normalize_measurements(measurements)
        event_polarities = self._normalize_event_polarities(
            event_polarities,
            measurements.shape[0],
        )
        event_signed_normal_flows = self._normalize_event_signed_normal_flows(
            event_signed_normal_flows,
            measurements.shape[0],
        )

        velocity = (
            self._get_event_velocity(event_velocity)
            if event_signed_normal_flows is None
            else None
        )
        (
            signed_flows,
            activities,
            resolved_polarity_contrast_sign,
            polarity_consistencies,
            polarity_weights,
            measurement_weights,
            active_measurement_mask,
        ) = self._event_measurement_weights(
            measurements,
            velocity,
            event_polarities,
            event_activity_floor,
            inactive_activity_threshold,
            polarity_mismatch_weight,
            polarity_contrast_sign,
            event_signed_normal_flows=event_signed_normal_flows,
        )

        self.last_event_activities = array(activities)
        self.last_event_signed_normal_flows = array(signed_flows)
        self.last_polarity_contrast_sign = resolved_polarity_contrast_sign
        if event_polarities is None or resolved_polarity_contrast_sign is None:
            self.last_event_polarity_consistencies = None
            self.last_event_polarity_weights = None
        else:
            self.last_event_polarity_consistencies = polarity_consistencies
            self.last_event_polarity_weights = array(polarity_weights)

        super().update(
            measurements,
            R=R,
            s_hat=s_hat,
            sigma_squared_s=sigma_squared_s,
            measurement_weights=measurement_weights,
            active_measurement_mask=active_measurement_mask,
        )
        self.last_active_measurement_indices = [
            index
            for index, is_active in enumerate(active_measurement_mask)
            if bool(is_active)
        ]


DVSSCGPTracker = DVSFullSCGPTracker
