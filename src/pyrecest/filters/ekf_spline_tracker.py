# pylint: disable=duplicate-code,no-member,no-name-in-module,too-many-lines
from pyrecest.backend import (
    abs,
    all,
    array,
    concatenate,
    cos,
    diag,
    eye,
    linalg,
    linspace,
    outer,
    sin,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


# pylint: disable=too-many-instance-attributes
class EKFSplineTracker(AbstractExtendedObjectTracker):
    """EKF tracker for a 2-D extended object with a closed quadratic spline extent.

    The state is ``[x, y, orientation, speed, turn_rate, scale_x, scale_y]``.
    The extent is represented by fixed body-frame spline control points and two
    estimated scale factors. Measurements are associated to the closest point on
    the currently predicted closed spline, then corrected with an EKF update.
    """

    def _initialize_extended_object_tracker(
        self,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        """Initialize shared extended-object tracker logging state."""
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        control_points=None,
        kinematic_state=None,
        scale_state=None,
        covariance=None,
        process_noise=None,
        measurement_noise=None,
        dt=1.0,
        acceleration_variance=0.0,
        turn_rate_variance=0.0,
        scale_process_noise=0.0,
        scale_correction=True,
        orientation_correction=True,
        finite_difference_step=1e-5,
        closest_point_grid_size=11,
        closest_point_iterations=8,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.measurement_dim = 2
        self.kinematic_dim = 5
        self.state_dim = 7
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

        kinematic_state = self._normalize_kinematic_state(kinematic_state)
        scale_state = self._normalize_scale_state(scale_state)
        self.state = concatenate([kinematic_state, scale_state])

        if covariance is None:
            covariance = diag(array([0.2, 0.2, 0.02, 0.05, 0.02, 0.1, 0.1]))
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
        self.scale_process_noise = float(scale_process_noise)
        self.scale_correction = bool(scale_correction)
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

    @staticmethod
    def default_control_points():
        """Return a rounded-rectangle control polygon used by the toolbox demo."""
        return array(
            [
                [2.5, 0.0],
                [2.5, 1.0],
                [0.0, 1.0],
                [-2.5, 1.0],
                [-2.5, 0.0],
                [-2.5, -1.0],
                [0.0, -1.0],
                [2.5, -1.0],
            ]
        )

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _as_covariance_matrix(
        cls,
        value,
        dim,
        name,
        require_positive_semidefinite=True,
    ):
        matrix = array(value)
        if matrix.ndim == 0:
            matrix = matrix * eye(dim)
        elif matrix.ndim == 1:
            if matrix.shape[0] != dim:
                raise ValueError(f"{name} vector must have length {dim}")
            matrix = diag(matrix)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        matrix = cls._symmetrize(matrix)
        if require_positive_semidefinite and not all(
            linalg.eigvalsh(matrix) >= -1e-12
        ):
            raise ValueError(f"{name} must be positive semidefinite")
        return matrix

    @staticmethod
    def _validate_control_points(control_points):
        control_points = array(control_points)
        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError("control_points must have shape (n_points, 2)")
        if control_points.shape[0] < 3:
            raise ValueError("At least three spline control points are required")
        return control_points

    @staticmethod
    def _normalize_kinematic_state(kinematic_state):
        if kinematic_state is None:
            return zeros(5)
        kinematic_state = array(kinematic_state)
        if kinematic_state.ndim != 1:
            raise ValueError("kinematic_state must be one-dimensional")
        if kinematic_state.shape[0] == 2:
            return concatenate([kinematic_state, zeros(3)])
        if kinematic_state.shape[0] == 3:
            return concatenate([kinematic_state, zeros(2)])
        if kinematic_state.shape[0] != 5:
            raise ValueError("kinematic_state must have length 2, 3, or 5")
        return kinematic_state

    @staticmethod
    def _normalize_scale_state(scale_state):
        if scale_state is None:
            scale_state = array([1.0, 1.0])
        scale_state = array(scale_state)
        if scale_state.shape != (2,):
            raise ValueError("scale_state must have shape (2,)")
        if float(scale_state[0]) <= 0.0 or float(scale_state[1]) <= 0.0:
            raise ValueError("scale_state entries must be positive")
        return scale_state

    def _sync_state_views(self):
        self.kinematic_state = self.state[: self.kinematic_dim]
        self.scale_state = self.state[-2:]

    @staticmethod
    def _rotation_matrix(orientation):
        return array(
            [
                [cos(orientation), -sin(orientation)],
                [sin(orientation), cos(orientation)],
            ]
        )

    def _segment_control_points(self, segment_index):
        return array(
            [
                self.control_points[(segment_index + offset) % self.num_control_points]
                for offset in range(3)
            ]
        )

    def _segment_coefficients(self, segment_index):
        return self._spline_basis @ self._segment_control_points(segment_index)

    @staticmethod
    def _evaluate_coefficients(coefficients, tau):
        tau_vector = array([tau**2, tau, 1.0])
        return tau_vector @ coefficients

    def _evaluate_segment(self, segment_index, tau):
        return self._evaluate_coefficients(
            self._segment_coefficients(segment_index),
            tau,
        )

    def _compute_segment_anchors(self):
        return array(
            [
                self._evaluate_segment(segment_index, 0.5)
                for segment_index in range(self.num_control_points)
            ]
        )

    @staticmethod
    def _safe_scale(value):
        value = float(value)
        if abs(value) < 1e-9:
            return 1e-9
        return value

    def _scaled_body_measurement(self, body_measurement, scales):
        safe_scales = array([self._safe_scale(scales[0]), self._safe_scale(scales[1])])
        return body_measurement / safe_scales

    def _select_segment(self, scaled_body_measurement):
        measurement_norm = float(linalg.norm(scaled_body_measurement))
        if measurement_norm <= 1e-12:
            return 0
        direction = scaled_body_measurement / measurement_norm
        best_index = 0
        best_score = -1e300
        for segment_index in range(self.num_control_points):
            anchor = self._segment_anchors[segment_index]
            anchor_norm = float(linalg.norm(anchor))
            if anchor_norm <= 1e-12:
                continue
            score = float((anchor / anchor_norm) @ direction)
            if score > best_score:
                best_score = score
                best_index = segment_index
        return best_index

    def _closest_tau_on_segment(self, segment_index, scaled_body_measurement):
        coefficients = self._segment_coefficients(segment_index)
        best_tau = 0.0
        best_distance = None
        for grid_index in range(self.closest_point_grid_size + 1):
            tau = grid_index / self.closest_point_grid_size
            diff = (
                self._evaluate_coefficients(coefficients, tau)
                - scaled_body_measurement
            )
            distance = float(diff @ diff)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_tau = tau

        tau = best_tau
        second_derivative = 2.0 * coefficients[0]
        for _ in range(self.closest_point_iterations):
            point = self._evaluate_coefficients(coefficients, tau)
            first_derivative = 2.0 * tau * coefficients[0] + coefficients[1]
            residual = point - scaled_body_measurement
            gradient = 2.0 * (first_derivative @ residual)
            curvature = 2.0 * (
                first_derivative @ first_derivative + residual @ second_derivative
            )
            if abs(float(curvature)) <= 1e-12:
                break
            tau = max(0.0, min(1.0, tau - float(gradient / curvature)))
        return tau

    def _project_measurement_to_body_spline(self, body_measurement, scales):
        scaled_body_measurement = self._scaled_body_measurement(
            body_measurement,
            scales,
        )
        segment_index = self._select_segment(scaled_body_measurement)
        tau = self._closest_tau_on_segment(segment_index, scaled_body_measurement)
        return self._evaluate_segment(segment_index, tau), segment_index, tau

    def _predict_measurement_from_state(self, state, measurement):
        position = state[:2]
        orientation = state[2]
        scales = state[-2:]
        rotation_matrix = self._rotation_matrix(orientation)
        body_measurement = rotation_matrix.T @ (measurement - position)
        body_spline_point, _, _ = self._project_measurement_to_body_spline(
            body_measurement,
            scales,
        )
        scaled_body_spline_point = body_spline_point * scales
        return position + rotation_matrix @ scaled_body_spline_point

    def _predict_measurements_from_state(self, state, measurements):
        return concatenate(
            [
                self._predict_measurement_from_state(state, measurement)
                for measurement in measurements
            ]
        )

    def _finite_difference_jacobian(self, function, point):
        point = array(point)
        base_value = function(point)
        jacobian_columns = []
        for dim_index in range(point.shape[0]):
            step = self.finite_difference_step * (1.0 + abs(float(point[dim_index])))
            perturbation = zeros(point.shape[0])
            perturbation[dim_index] = step
            jacobian_columns.append(
                (function(point + perturbation) - function(point - perturbation))
                / (2.0 * step)
            )
        if not jacobian_columns:
            return zeros((base_value.shape[0], 0))
        return array(jacobian_columns).T

    def _transition_state(self, state, dt):
        transitioned = array(state)
        orientation = state[2]
        speed = state[3]
        turn_rate = state[4]
        transitioned[0] = state[0] + dt * speed * cos(orientation)
        transitioned[1] = state[1] + dt * speed * sin(orientation)
        transitioned[2] = state[2] + dt * turn_rate
        return transitioned

    def _process_noise(self, dt):
        process_noise = array(self.process_noise)
        if self.acceleration_variance > 0.0:
            orientation = self.state[2]
            position_direction = 0.5 * dt**2 * array(
                [cos(orientation), sin(orientation)]
            )
            process_noise[0:2, 0:2] = process_noise[0:2, 0:2] + (
                self.acceleration_variance
                * outer(position_direction, position_direction)
            )
            process_noise[0:2, 3] = process_noise[0:2, 3] + (
                self.acceleration_variance * position_direction * dt
            )
            process_noise[3, 0:2] = process_noise[0:2, 3]
            process_noise[3, 3] = process_noise[3, 3] + (
                self.acceleration_variance * dt**2
            )
        if self.turn_rate_variance > 0.0:
            process_noise[2, 2] = process_noise[2, 2] + (
                self.turn_rate_variance * dt**3 / 3.0
            )
            process_noise[2, 4] = process_noise[2, 4] + (
                self.turn_rate_variance * dt**2 / 2.0
            )
            process_noise[4, 2] = process_noise[2, 4]
            process_noise[4, 4] = process_noise[4, 4] + (
                self.turn_rate_variance * dt
            )
        if self.scale_process_noise > 0.0:
            process_noise[5, 5] = process_noise[5, 5] + self.scale_process_noise * dt
            process_noise[6, 6] = process_noise[6, 6] + self.scale_process_noise * dt
        return self._symmetrize(process_noise)

    def predict(self, dt=None, process_noise=None):
        if dt is None:
            dt = self.dt
        else:
            dt = float(dt)

        def transition_function(state):
            return self._transition_state(state, dt)

        transition_jacobian = self._finite_difference_jacobian(
            transition_function,
            self.state,
        )
        self.state = transition_function(self.state)
        if process_noise is None:
            process_noise = self._process_noise(dt)
        else:
            process_noise = self._as_covariance_matrix(
                process_noise,
                self.state_dim,
                "process_noise",
                require_positive_semidefinite=False,
            )
        self.covariance = self._symmetrize(
            transition_jacobian @ self.covariance @ transition_jacobian.T
            + process_noise
        )
        self._sync_state_views()

        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def _normalize_measurements(self, measurements):
        measurements = array(measurements)
        if measurements.ndim == 1:
            if measurements.shape[0] != self.measurement_dim:
                raise ValueError("A single measurement must be two-dimensional")
            return measurements.reshape((1, self.measurement_dim))
        if measurements.ndim != 2:
            raise ValueError("measurements must be a vector or a two-dimensional array")
        if measurements.shape[1] == self.measurement_dim:
            return measurements
        if measurements.shape[0] == self.measurement_dim:
            return measurements.T
        raise ValueError(
            "measurements must have shape (2, n_measurements) or "
            "(n_measurements, 2)"
        )

    def update(self, measurements, R=None):  # pylint: disable=too-many-locals
        measurements = self._normalize_measurements(measurements)
        if R is None:
            measurement_noise = self.measurement_noise
        else:
            measurement_noise = self._as_covariance_matrix(
                R,
                self.measurement_dim,
                "R",
                require_positive_semidefinite=False,
            )

        def measurement_function(state):
            return self._predict_measurements_from_state(
                state,
                measurements,
            )

        predicted_measurements = measurement_function(self.state)
        measurement_jacobian = self._finite_difference_jacobian(
            measurement_function,
            self.state,
        )
        if not self.orientation_correction:
            measurement_jacobian[:, 2] = 0.0
        if not self.scale_correction:
            measurement_jacobian[:, -2:] = 0.0

        stacked_measurements = concatenate(list(measurements))
        residual = stacked_measurements - predicted_measurements
        block_measurement_noise = linalg.block_diag(
            *[measurement_noise for _ in range(measurements.shape[0])]
        )
        innovation_covariance = self._symmetrize(
            measurement_jacobian @ self.covariance @ measurement_jacobian.T
            + block_measurement_noise
        )
        cross_covariance = self.covariance @ measurement_jacobian.T
        gain = linalg.solve(innovation_covariance.T, cross_covariance.T).T
        self.state = self.state + gain @ residual
        self.covariance = self._symmetrize(
            self.covariance - gain @ innovation_covariance @ gain.T
        )
        self._sync_state_views()
        self.last_quadratic_form = residual @ linalg.solve(
            innovation_covariance,
            residual,
        )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def get_scaled_control_points(self, global_frame=False):
        scaled_control_points = self.control_points * self.scale_state
        if not global_frame:
            return scaled_control_points
        rotation_matrix = self._rotation_matrix(self.state[2])
        return (rotation_matrix @ scaled_control_points.T).T + self.state[:2]

    def get_point_estimate(self):
        return self.state

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        scaled_control_points = self.get_scaled_control_points()
        if flatten_matrix:
            return scaled_control_points.flatten()
        return scaled_control_points

    def get_contour_points(self, n=100, scaling_factor=1.0):
        if n <= 0:
            raise ValueError("n must be positive")
        segment_coordinates = linspace(
            0.0,
            float(self.num_control_points),
            n,
            endpoint=False,
        )
        rotation_matrix = self._rotation_matrix(self.state[2])
        contour_points = []
        for segment_coordinate in segment_coordinates:
            segment_index = int(float(segment_coordinate)) % self.num_control_points
            tau = float(segment_coordinate) - int(float(segment_coordinate))
            body_point = self._evaluate_segment(segment_index, tau)
            scaled_body_point = scaling_factor * body_point * self.scale_state
            contour_points.append(self.state[:2] + rotation_matrix @ scaled_body_point)
        return array(contour_points)

    def get_bounding_box(self, n=100):
        contour_points = self.get_contour_points(n)
        xs = [float(point[0]) for point in contour_points]
        ys = [float(point[1]) for point in contour_points]
        min_xy = array([min(xs), min(ys)])
        max_xy = array([max(xs), max(ys)])
        return {
            "center_xy": 0.5 * (min_xy + max_xy),
            "dimension": max_xy - min_xy,
            "orientation": self.state[2],
        }


EkfSplineTracker = EKFSplineTracker
