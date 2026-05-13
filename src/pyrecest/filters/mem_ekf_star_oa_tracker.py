from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,duplicate-code,too-many-arguments
from pyrecest.backend import arctan2, array, cos, diag, sin

from .mem_ekf_star_tracker import MEMEKFStarTracker


class MEMEKFStarOATracker(MEMEKFStarTracker):
    """MEM-EKF* with the orientation approximation used by MEM-EKF*-OA.

    The shape convention is the same as :class:`MEMEKFStarTracker`:
    ``shape_state = [orientation, semi_axis_1, semi_axis_2]``.  In the OA
    variant, however, the orientation used in the multiplicative-error matrix
    and in the MEM-EKF* row Jacobians is not the stored shape orientation. It is
    computed from the kinematic velocity as

    ``orientation = atan2(v_y, v_x) + orientation_offset``.

    Matching the original ``al_approx=True`` implementation, the Jacobian
    columns with respect to the explicit orientation state are set to zero. The
    stored orientation mean is realigned to the velocity heading after
    initialization, prediction, and every single-measurement update; the shape
    covariance is kept in the original MEM-EKF* shape coordinates.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        *args,
        velocity_indices=(2, 3),
        orientation_offset=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.velocity_indices = self._normalize_velocity_indices(velocity_indices)
        self.orientation_offset = float(orientation_offset)
        self._align_shape_orientation_with_velocity()

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

    def _velocity_orientation(self):
        velocity_x_index, velocity_y_index = self.velocity_indices
        velocity_x = self.kinematic_state[velocity_x_index]
        velocity_y = self.kinematic_state[velocity_y_index]
        return arctan2(velocity_y, velocity_x) + self.orientation_offset

    def _align_shape_orientation_with_velocity(self):
        self.shape_state = array(
            [self._velocity_orientation(), self.shape_state[1], self.shape_state[2]]
        )

    def _extent_transform(self):
        orientation = self._velocity_orientation()
        semi_axis_1 = self.shape_state[1]
        semi_axis_2 = self.shape_state[2]
        rotation_matrix = array(
            [
                [cos(orientation), -sin(orientation)],
                [sin(orientation), cos(orientation)],
            ]
        )
        return rotation_matrix @ diag(array([semi_axis_1, semi_axis_2]))

    def _extent_row_jacobians(self):
        orientation = self._velocity_orientation()

        # Same row-Jacobian ordering as MEMEKFStarTracker, but with the
        # derivative w.r.t. the explicit orientation state forced to zero. This
        # is the MEM-EKF*-OA / al_approx=True approximation.
        first_row_jacobian = array(
            [
                [0.0, cos(orientation), 0.0],
                [0.0, 0.0, -sin(orientation)],
            ]
        )
        second_row_jacobian = array(
            [
                [0.0, sin(orientation), 0.0],
                [0.0, 0.0, cos(orientation)],
            ]
        )
        return first_row_jacobian, second_row_jacobian

    # pylint: disable=too-many-positional-arguments
    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_system_matrix=None,
        shape_sys_noise=None,
    ):
        """Predict one step and store the velocity-derived orientation mean."""
        log_prior_estimates = self.log_prior_estimates
        log_prior_extents = self.log_prior_extents
        self.log_prior_estimates = False
        self.log_prior_extents = False
        try:
            super().predict_linear(
                system_matrix,
                sys_noise=sys_noise,
                inputs=inputs,
                shape_system_matrix=shape_system_matrix,
                shape_sys_noise=shape_sys_noise,
            )
        finally:
            self.log_prior_estimates = log_prior_estimates
            self.log_prior_extents = log_prior_extents

        self._align_shape_orientation_with_velocity()
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def _update_single_measurement(
        self,
        measurement,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
    ):
        super()._update_single_measurement(
            measurement,
            measurement_matrix,
            meas_noise_cov,
            multiplicative_noise_cov,
        )
        self._align_shape_orientation_with_velocity()


MemEkfStarOATracker = MEMEKFStarOATracker
VelocityAlignedMEMEKFStarTracker = MEMEKFStarOATracker
VelocityAlignedMemEkfStarTracker = MEMEKFStarOATracker
