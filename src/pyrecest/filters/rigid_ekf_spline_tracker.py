# pylint: disable=duplicate-code,no-member,no-name-in-module,super-init-not-called,too-many-instance-attributes
from pyrecest.backend import array, diag, eye, zeros

from .ekf_spline_tracker import EKFSplineTracker


class RigidEKFSplineTracker(EKFSplineTracker):
    """EKF spline tracker with a fixed body-frame spline extent.

    The state is ``[x, y, orientation, speed, turn_rate]``. In contrast to
    :class:`EKFSplineTracker`, this tracker does not estimate scale factors;
    measurements are projected onto the unscaled closed spline contour.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        control_points=None,
        kinematic_state=None,
        covariance=None,
        process_noise=None,
        measurement_noise=None,
        dt=1.0,
        acceleration_variance=0.0,
        turn_rate_variance=0.0,
        orientation_correction=True,
        finite_difference_step=1e-5,
        closest_point_grid_size=11,
        closest_point_iterations=8,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        self._initialize_extended_object_tracker(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.measurement_dim = 2
        self.kinematic_dim = 5
        self.state_dim = 5
        self._spline_basis = array(
            [
                [0.5, -1.0, 0.5],
                [-1.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        if control_points is None:
            control_points = self.default_control_points()
        self.control_points = self._validate_control_points(control_points)
        self.num_control_points = self.control_points.shape[0]
        self._segment_anchors = self._compute_segment_anchors()

        self.state = array(self._normalize_kinematic_state(kinematic_state))
        self._unit_scale_state = array([1.0, 1.0])

        if covariance is None:
            covariance = diag(array([0.2, 0.2, 0.02, 0.05, 0.02]))
        self.covariance = self._as_covariance_matrix(
            covariance,
            self.state_dim,
            "covariance",
        )
        if process_noise is None:
            process_noise = zeros((self.state_dim, self.state_dim))
        self.process_noise = self._as_covariance_matrix(
            process_noise,
            self.state_dim,
            "process_noise",
            require_positive_semidefinite=False,
        )
        if measurement_noise is None:
            measurement_noise = 0.05 * eye(self.measurement_dim)
        self.measurement_noise = self._as_covariance_matrix(
            measurement_noise,
            self.measurement_dim,
            "measurement_noise",
            require_positive_semidefinite=False,
        )

        self.dt = float(dt)
        self.acceleration_variance = float(acceleration_variance)
        self.turn_rate_variance = float(turn_rate_variance)
        self.scale_process_noise = 0.0
        # Keep the inherited EKF update from treating velocity columns as scales.
        self.scale_correction = True
        self.orientation_correction = bool(orientation_correction)
        self.finite_difference_step = float(finite_difference_step)
        self.closest_point_grid_size = int(closest_point_grid_size)
        self.closest_point_iterations = int(closest_point_iterations)
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        if self.closest_point_grid_size <= 1:
            raise ValueError("closest_point_grid_size must be greater than one")
        if self.closest_point_iterations < 0:
            raise ValueError("closest_point_iterations must be non-negative")
        self.last_quadratic_form = None
        self._sync_state_views()

    def _sync_state_views(self):
        self.kinematic_state = self.state
        self.scale_state = self._unit_scale_state

    def _predict_measurement_from_state(self, state, measurement):
        position = state[:2]
        orientation = state[2]
        rotation_matrix = self._rotation_matrix(orientation)
        body_measurement = rotation_matrix.T @ (measurement - position)
        body_spline_point, _, _ = self._project_measurement_to_body_spline(
            body_measurement,
            self._unit_scale_state,
        )
        return position + rotation_matrix @ body_spline_point

    def get_scaled_control_points(self, global_frame=False):
        control_points = array(self.control_points)
        if not global_frame:
            return control_points
        rotation_matrix = self._rotation_matrix(self.state[2])
        return (rotation_matrix @ control_points.T).T + self.state[:2]

    def get_point_estimate(self):
        return self.state

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        control_points = self.get_scaled_control_points()
        if flatten_matrix:
            return control_points.flatten()
        return control_points


RigidEkfSplineTracker = RigidEKFSplineTracker
