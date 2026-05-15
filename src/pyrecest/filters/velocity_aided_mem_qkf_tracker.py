from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
from pyrecest.backend import arctan2, array, cos, maximum, sin

from .mem_qkf_tracker import MEMQKFTracker


class VelocityAidedMEMQKFTracker(MEMQKFTracker):
    """MEM-QKF with a soft axial velocity-heading pseudo-measurement.

    This tracker keeps the ordinary MEM-QKF shape representation
    ``[orientation, semi_axis_1, semi_axis_2]`` with a local scalar Gaussian
    moment approximation for the orientation. Unlike
    :class:`VelocityAlignedMEMQKFTracker`, it does not overwrite the orientation
    with the kinematic heading. Instead, when the estimated speed is above
    ``speed_threshold``, it fuses one additional axial pseudo-measurement

    ``psi = atan2(v_y, v_x) + orientation_offset``

    using a local Kalman update with a pi-periodic residual. This is equivalent
    to a local wrapped-normal likelihood on ellipse orientation, where
    ``theta`` and ``theta + pi`` denote the same physical extent. The heading
    variance is derived from the velocity covariance by the first-order delta
    method and inflated by ``heading_noise_variance``.

    The heading pseudo-measurement is attempted once per measurement scan before
    the per-point shape updates. This avoids counting the same velocity-heading
    information once per target-originated point.

    Parameters added by this subclass
    ---------------------------------
    velocity_indices : tuple[int, int], default=(2, 3)
        Indices of ``v_x`` and ``v_y`` in the Euclidean kinematic state.
    speed_threshold : float, default=1e-9
        Minimum speed required before the heading pseudo-measurement is used.
    orientation_offset : float, default=0.0
        Constant offset from velocity heading to ellipse orientation, useful for
        fixed sideslip or convention differences.
    heading_noise_variance : float, default=0.0
        Additional variance of the soft velocity-heading relation.
    minimum_heading_variance : float, default=1e-12
        Lower bound for the heading pseudo-measurement variance.
    apply_heading_on_prediction : bool, default=False
        If true, also fuse the heading pseudo-measurement after predictions.
        The default keeps the pseudo-measurement tied to measurement updates so
        it is not repeatedly counted when several predictions occur without new
        measurements.
    use_heading_constraint : bool, default=True
        Enables or disables the velocity-heading pseudo-measurement.
    """

    def __init__(
        self,
        *args,
        velocity_indices=(2, 3),
        speed_threshold=1e-9,
        orientation_offset=0.0,
        heading_noise_variance=0.0,
        minimum_heading_variance=1e-12,
        apply_heading_on_prediction=False,
        use_heading_constraint=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.velocity_indices = self._normalize_velocity_indices(velocity_indices)
        self.speed_threshold = float(speed_threshold)
        if self.speed_threshold < 0.0:
            raise ValueError("speed_threshold must be non-negative")

        self.orientation_offset = float(orientation_offset)
        self.heading_noise_variance = float(heading_noise_variance)
        if self.heading_noise_variance < 0.0:
            raise ValueError("heading_noise_variance must be non-negative")

        self.minimum_heading_variance = float(minimum_heading_variance)
        if self.minimum_heading_variance <= 0.0:
            raise ValueError("minimum_heading_variance must be positive")

        self.apply_heading_on_prediction = bool(apply_heading_on_prediction)
        self.use_heading_constraint = bool(use_heading_constraint)
        self._heading_update_pending = False

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

    @staticmethod
    def _wrap_axial_residual(angle):
        """Wrap an ellipse-orientation residual to [-pi/2, pi/2)."""
        return 0.5 * arctan2(sin(2.0 * angle), cos(2.0 * angle))

    def _heading_moments_from_velocity(self, kinematic_state, covariance):
        if not self.use_heading_constraint:
            return None

        velocity_x_index, velocity_y_index = self.velocity_indices
        velocity_x = kinematic_state[velocity_x_index]
        velocity_y = kinematic_state[velocity_y_index]
        speed_squared = velocity_x**2 + velocity_y**2
        if float(speed_squared) <= self.speed_threshold**2:
            return None

        heading = arctan2(velocity_y, velocity_x) + self.orientation_offset

        # Jacobian of atan2(v_y, v_x) with respect to the full kinematic state.
        jacobian_entries = [0.0] * self.kinematic_state.shape[0]
        jacobian_entries[velocity_x_index] = -velocity_y / speed_squared
        jacobian_entries[velocity_y_index] = velocity_x / speed_squared
        heading_jacobian = array(jacobian_entries)
        heading_variance = (
            heading_jacobian @ covariance @ heading_jacobian.T
            + self.heading_noise_variance
        )
        heading_variance = maximum(heading_variance, self.minimum_heading_variance)
        return heading, heading_variance

    def _fuse_orientation_with_heading_values(
        self,
        kinematic_state,
        covariance,
        orientation,
        orientation_variance,
    ):
        heading_moments = self._heading_moments_from_velocity(
            kinematic_state,
            covariance,
        )
        if heading_moments is None:
            return orientation, orientation_variance, False

        heading, heading_variance = heading_moments
        innovation = self._wrap_axial_residual(heading - orientation)
        innovation_variance = self._regularize_variance(
            orientation_variance + heading_variance
        )
        gain = orientation_variance / innovation_variance
        posterior_orientation = orientation + gain * innovation
        posterior_orientation_variance = orientation_variance - gain * orientation_variance
        posterior_orientation_variance = self._regularize_variance(
            posterior_orientation_variance
        )
        return posterior_orientation, posterior_orientation_variance, True

    def _update_orientation_from_heading_state(self):
        orientation, orientation_variance, used_heading = (
            self._fuse_orientation_with_heading_values(
                self.kinematic_state,
                self.covariance,
                self.shape_state[0],
                self.orientation_variance,
            )
        )
        if not used_heading:
            return False
        self.shape_state = array(
            [orientation, self.shape_state[1], self.shape_state[2]]
        )
        self.shape_covariance = self._decoupled_shape_covariance(
            orientation_variance,
            self.axis_covariance,
        )
        return True

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_system_matrix=None,
        shape_sys_noise=None,
    ):
        """Predict one step and optionally apply the soft heading constraint."""
        super().predict_linear(
            system_matrix,
            sys_noise=sys_noise,
            inputs=inputs,
            shape_system_matrix=shape_system_matrix,
            shape_sys_noise=shape_sys_noise,
        )
        if self.apply_heading_on_prediction:
            self._update_orientation_from_heading_state()

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
        use_heading_constraint=None,
    ):
        """Update and fuse the velocity-heading pseudo-measurement once per scan."""
        old_use_heading_constraint = self.use_heading_constraint
        if use_heading_constraint is not None:
            self.use_heading_constraint = bool(use_heading_constraint)
        self._heading_update_pending = True
        try:
            super().update(
                measurements,
                meas_mat=meas_mat,
                meas_noise_cov=meas_noise_cov,
                multiplicative_noise_cov=multiplicative_noise_cov,
            )
        finally:
            self._heading_update_pending = False
            self.use_heading_constraint = old_use_heading_constraint

    def _update_single_measurement_qkf(
        self,
        measurement,
        center_estimate,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
        shape_measurement_covariance,
        update_kinematics=True,
    ):
        orientation = self.shape_state[0]
        semi_axes = self.shape_state[1:]
        orientation_variance = self.orientation_variance
        axis_covariance = self.axis_covariance
        axis_update_orientation = self.shape_state[0]
        use_posterior_orientation_for_axis_update = False

        kinematic_state = self.kinematic_state
        covariance = self.covariance
        if update_kinematics:
            kinematic_state, covariance = self._kinematic_update(
                measurement,
                measurement_matrix,
                meas_noise_cov,
                multiplicative_noise_cov,
            )
        covariance = self._project_symmetric_covariance(covariance)

        if self._heading_update_pending:
            (
                orientation,
                orientation_variance,
                used_heading,
            ) = self._fuse_orientation_with_heading_values(
                kinematic_state,
                covariance,
                orientation,
                orientation_variance,
            )
            self._heading_update_pending = False
            if used_heading:
                axis_update_orientation = orientation
                use_posterior_orientation_for_axis_update = True

        orientation, orientation_variance = self._orientation_update(
            measurement,
            center_estimate,
            orientation,
            semi_axes,
            orientation_variance,
            multiplicative_noise_cov,
            shape_measurement_covariance,
        )
        if use_posterior_orientation_for_axis_update:
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


VelocityAidedMemQkfTracker = VelocityAidedMEMQKFTracker
