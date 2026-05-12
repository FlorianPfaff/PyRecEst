from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,too-many-arguments
from pyrecest.backend import arctan2, array, maximum

from .mem_qkf_tracker import MEMQKFTracker


class VelocityLockedMEMQKFTracker(MEMQKFTracker):
    """MEM-QKF whose moving-mode orientation is locked to kinematic velocity.

    This tracker is a matched-model variant for elongated objects whose
    longitudinal axis is expected to align with the heading. When the estimated
    speed is above ``speed_threshold``, the ellipse orientation is not estimated
    from the quadratic orientation pseudo-measurement. Instead, it is computed
    from the kinematic state as

    ``theta = atan2(v_y, v_x) + orientation_offset``.

    The orientation variance is propagated from the kinematic covariance by a
    first-order delta method using the Jacobian of ``atan2(v_y, v_x)``. The
    semi-axis update remains the MEM-QKF quadratic update, but it is performed in
    the velocity-aligned body frame. If the speed falls below the threshold, the
    class falls back to the standard MEM-QKF update so stationary targets can
    still estimate orientation from the measurement cloud.

    Parameters added by this subclass
    ---------------------------------
    velocity_indices : tuple[int, int], default=(2, 3)
        Indices of ``v_x`` and ``v_y`` in the Euclidean kinematic state.
    speed_threshold : float, default=1e-9
        Minimum speed required before the heading lock is applied.
    orientation_offset : float, default=0.0
        Constant offset from velocity heading to ellipse orientation, useful for
        modeling a fixed sideslip or a convention difference.
    sideslip_variance : float, default=0.0
        Extra variance added to the heading-derived orientation variance.
    minimum_orientation_variance : float, default=1e-12
        Lower bound for the locked orientation variance.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        *args,
        velocity_indices=(2, 3),
        speed_threshold=1e-9,
        orientation_offset=0.0,
        sideslip_variance=0.0,
        minimum_orientation_variance=1e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.velocity_indices = self._normalize_velocity_indices(velocity_indices)
        self.speed_threshold = float(speed_threshold)
        if self.speed_threshold < 0.0:
            raise ValueError("speed_threshold must be non-negative")

        self.orientation_offset = float(orientation_offset)
        self.sideslip_variance = float(sideslip_variance)
        if self.sideslip_variance < 0.0:
            raise ValueError("sideslip_variance must be non-negative")

        self.minimum_orientation_variance = float(minimum_orientation_variance)
        if self.minimum_orientation_variance < 0.0:
            raise ValueError("minimum_orientation_variance must be non-negative")

        self._lock_orientation_to_velocity_if_moving()

    def _normalize_velocity_indices(self, velocity_indices):
        if len(velocity_indices) != 2:
            raise ValueError("velocity_indices must contain exactly two entries")
        state_dim = self.kinematic_state.shape[0]
        normalized = []
        for index in velocity_indices:
            index = int(index)
            if index < 0:
                index += state_dim
            if index < 0 or index >= state_dim:
                raise ValueError(
                    "velocity index out of bounds for kinematic state dimension "
                    f"{state_dim}: {velocity_indices}"
                )
            normalized.append(index)
        if normalized[0] == normalized[1]:
            raise ValueError("velocity_indices must refer to two distinct states")
        return tuple(normalized)

    def _heading_moments_from_velocity(self, kinematic_state, covariance):
        velocity_x_index, velocity_y_index = self.velocity_indices
        velocity_x = kinematic_state[velocity_x_index]
        velocity_y = kinematic_state[velocity_y_index]
        speed_squared = velocity_x**2 + velocity_y**2
        if float(speed_squared) <= self.speed_threshold**2:
            return None

        orientation = arctan2(velocity_y, velocity_x) + self.orientation_offset

        # Jacobian of atan2(v_y, v_x) with respect to the full kinematic state.
        jacobian_entries = [0.0] * self.kinematic_state.shape[0]
        jacobian_entries[velocity_x_index] = -velocity_y / speed_squared
        jacobian_entries[velocity_y_index] = velocity_x / speed_squared
        heading_jacobian = array(jacobian_entries)
        orientation_variance = (
            heading_jacobian @ covariance @ heading_jacobian.T + self.sideslip_variance
        )
        orientation_variance = maximum(
            orientation_variance,
            self.minimum_orientation_variance,
        )
        return orientation, orientation_variance

    def _lock_orientation_to_velocity_if_moving(self):
        heading_moments = self._heading_moments_from_velocity(
            self.kinematic_state,
            self.covariance,
        )
        if heading_moments is None:
            return False
        orientation, orientation_variance = heading_moments
        self.shape_state = array(
            [orientation, self.shape_state[1], self.shape_state[2]]
        )
        self.shape_covariance = self._decoupled_shape_covariance(
            orientation_variance,
            self.axis_covariance,
        )
        return True

    # pylint: disable=too-many-positional-arguments
    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_system_matrix=None,
        shape_sys_noise=None,
    ):
        """Predict one step and relock the orientation whenever moving."""
        super().predict_linear(
            system_matrix,
            sys_noise=sys_noise,
            inputs=inputs,
            shape_system_matrix=shape_system_matrix,
            shape_sys_noise=shape_sys_noise,
        )
        self._lock_orientation_to_velocity_if_moving()

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        """Update and use heading-locked orientation for moving targets."""
        self._lock_orientation_to_velocity_if_moving()
        super().update(
            measurements,
            meas_mat=meas_mat,
            meas_noise_cov=meas_noise_cov,
            multiplicative_noise_cov=multiplicative_noise_cov,
        )
        self._lock_orientation_to_velocity_if_moving()

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _update_single_measurement_qkf(
        self,
        measurement,
        center_estimate,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
        shape_measurement_covariance,
    ):
        # Stationary/near-stationary fallback: keep the full MEM-QKF orientation
        # update so the filter remains usable for fixed targets.
        if not self._lock_orientation_to_velocity_if_moving():
            super()._update_single_measurement_qkf(
                measurement,
                center_estimate,
                measurement_matrix,
                meas_noise_cov,
                multiplicative_noise_cov,
                shape_measurement_covariance,
            )
            return

        semi_axes = self.shape_state[1:]
        axis_covariance = self.axis_covariance

        kinematic_state, covariance = self._kinematic_update(
            measurement,
            measurement_matrix,
            meas_noise_cov,
            multiplicative_noise_cov,
        )
        covariance = self._project_symmetric_covariance(covariance)
        heading_moments = self._heading_moments_from_velocity(
            kinematic_state,
            covariance,
        )
        if heading_moments is None:
            # The update can slow the target below threshold. Fall back to a
            # standard orientation update using the pre-update orientation state.
            orientation, orientation_variance = self._orientation_update(
                measurement,
                center_estimate,
                self.shape_state[0],
                semi_axes,
                self.orientation_variance,
                multiplicative_noise_cov,
                shape_measurement_covariance,
            )
            axis_update_orientation = self.shape_state[0]
        else:
            orientation, orientation_variance = heading_moments
            axis_update_orientation = orientation

        semi_axes, axis_covariance = self._axis_update(
            measurement,
            center_estimate,
            axis_update_orientation,
            semi_axes,
            axis_covariance,
            multiplicative_noise_cov,
            shape_measurement_covariance,
        )
        semi_axes, axis_covariance = self._canonicalize_axes_and_axis_covariance(
            semi_axes,
            axis_covariance,
        )

        self.kinematic_state = kinematic_state
        self.covariance = covariance
        self.shape_state = array([orientation, semi_axes[0], semi_axes[1]])
        self.shape_covariance = self._decoupled_shape_covariance(
            self._regularize_variance(orientation_variance),
            axis_covariance,
        )


VelocityLockedMemQkfTracker = VelocityLockedMEMQKFTracker
