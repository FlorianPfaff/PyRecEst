# pylint: disable=duplicate-code,invalid-name,no-member,no-name-in-module
# pylint: disable=redefined-builtin,too-many-lines,too-many-locals
from pyrecest.backend import (
    abs,
    all,
    arctan2,
    array,
    concatenate,
    cos,
    dot,
    exp,
    eye,
    hstack,
    isfinite,
    linalg,
    linspace,
    max,
    min,
    ndim,
    ones,
    outer,
    pi,
    reshape,
    sin,
    stack,
    vstack,
    zeros,
    zeros_like,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


def pol2cart(phi, r=1.0):
    if ndim(phi) > 1:
        r = reshape(r, (1, -1))
    return r * stack((cos(phi), sin(phi)))


def angle_between_two_vectors(x, y):
    dot_prod = dot(x, y)
    cross_prod = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    return -arctan2(cross_prod, dot_prod) % (2 * pi)


# pylint: disable=too-many-instance-attributes
class GPRHMTracker(AbstractExtendedObjectTracker):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        n_base_points,
        dimension: int = 2,
        velocities=False,
        kernel_params=(2.0, pi / 4),
        log_prior_estimates=True,
        log_posterior_estimates=True,
        log_prior_extents=True,
        log_posterior_extents=True,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )

        def sin_kernel(
            phi_1, phi_2, sigma_squared=kernel_params[0], kernel_width=kernel_params[1]
        ):
            d = min(array((abs(phi_1 - phi_2), 2 * pi - abs(phi_1 - phi_2))))
            exponent = 2 * sin((0.5) * d) ** 2 / kernel_width**2
            return sigma_squared * exp(-exponent)

        self.kernel = sin_kernel
        self.phi_pts = linspace(0.0, 2 * pi, n_base_points, endpoint=False)
        if dimension == 2 and not velocities:
            self.m = zeros(2)
            self.H = eye(2)
            self.C_m = 0.1 * eye(2)
        else:
            raise NotImplementedError(
                "Only 2-D scenarios without velocity estimation supported"
            )

        self.p = zeros_like(self.phi_pts)
        self.C_p = 0.1 * eye(self.phi_pts.shape[0])

        def K_fun(phi):
            return array([[self.kernel(phi, phi_n) for phi_n in self.phi_pts]])

        K_p = array(
            [
                [
                    self.kernel(self.phi_pts[i], self.phi_pts[j])
                    for i in range(len(self.phi_pts))
                ]
                for j in range(len(self.phi_pts))
            ]
        )
        self.A_fun = lambda phi: linalg.solve(K_p, K_fun(phi).T).T
        self.C_e_fun = (
            lambda phi: self.kernel(phi, phi)
            - K_fun(phi) @ linalg.solve(K_p, K_fun(phi).T).T
        )

    def get_point_estimate(self):
        # Return the state concatenated with the flattened extent
        return hstack([self.m, self.p.flatten()])

    def get_point_estimate_kinematics(self):
        # Return only the kinematic state
        return self.m

    def get_point_estimate_extent(self, flatten_matrix=False):
        # Return the extent matrix, optionally flattened
        return self.p.flatten() if flatten_matrix else self.p

    def get_extents_on_grid(self, n: int = 100):
        angles = linspace(0.0, 2 * pi, n, endpoint=False)
        extents = array([linalg.norm(self.A_fun(phi) @ self.p) for phi in angles])
        return extents

    def get_contour_points(self, n: int = 100):
        angles = linspace(0.0, 2 * pi, n, endpoint=False)
        star_point = self.H @ self.m
        extents = self.get_extents_on_grid(n)
        coords = pol2cart(angles, extents) + reshape(star_point, (-1, 1))
        return coords.T

    def update(
        self,
        z,
        R,
        # s_hat=1 and sigma_squared_s=0 means that measurements only come from the contour
        s_hat=1,
        sigma_squared_s=0,
    ):
        phi = angle_between_two_vectors(z - self.H @ self.m, pol2cart(array(0)))
        B_phi = s_hat * pol2cart(phi)[:, None] @ self.A_fun(phi)

        if sigma_squared_s == 0:  # Use simpler formula if sigma_squared_s is zero
            C_w = R
        else:
            C_w = (
                sigma_squared_s
                * (
                    (pol2cart(phi) @ self.A_fun(phi))
                    @ self.C_p
                    @ (pol2cart(phi) @ self.A_fun(phi)).T
                    + pol2cart(phi) @ self.C_e_fun(phi) @ pol2cart(phi).T
                )
                + R
            )

        # Compute covariance matrix for the measurement z
        C_z = B_phi @ self.C_p @ B_phi.T + self.H @ self.C_m @ self.H.T + C_w

        # Compute cross-covariance matrices
        C_mz = self.C_m @ self.H.T
        C_pz = self.C_p @ B_phi.T

        # Mean of the expected measurement
        z_bar = B_phi @ self.p + self.H @ self.m

        # Compute the vector difference between z and z_bar
        delta_z = z - z_bar

        # Solve for the Kalman gain for the state update
        K_m = linalg.solve(C_z, C_mz.T).T

        # Update the state vector and covariance matrix
        self.m = self.m + K_m @ delta_z
        self.C_m = self.C_m - K_m @ C_mz.T

        # Solve for the Kalman gain for the extent parameters update
        K_p = linalg.solve(C_z, C_pz.T).T

        # Update the extent parameters and covariance matrix
        self.p = self.p + K_p @ delta_z
        self.C_p = self.C_p - K_p @ C_pz.T

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class FullSCGPTracker(AbstractExtendedObjectTracker):
    """Full star-convex Gaussian-process tracker.

    The state is ``[x, y, orientation, velocity, turn_rate, f_1, ..., f_n]`` by
    default. Set ``velocities=False`` to use the reduced kinematic state
    ``[x, y, orientation]``. Unlike :class:`GPRHMTracker`, this variant keeps one
    joint covariance over kinematics and GP extent coefficients, so update and
    prediction steps can propagate kinematic-shape correlations.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_base_points,
        kinematic_state=None,
        kinematic_covariance=None,
        shape_state=None,
        shape_covariance=None,
        joint_covariance=None,
        velocities=True,
        kernel_params=(2.0, pi / 4),
        dt=1.0,
        sys_noise=None,
        acceleration_variance=0.0,
        extent_forgetting_rate=0.0,
        reference_extent=None,
        radial_noise_variance=0.0,
        measurement_noise=None,
        scale_mean=1.0,
        scale_variance=0.0,
        alpha=1.0,
        beta=2.0,
        kappa=2.0,
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
        self.phi_pts = linspace(0.0, 2 * pi, n_base_points, endpoint=False)
        self.kernel_params = kernel_params
        self._kernel_variance = kernel_params[0]
        self._kernel_width = kernel_params[1]
        self._k_uu = self._kernel_matrix(self.phi_pts, self.phi_pts)
        self._k_uu_inv = linalg.solve(self._k_uu, eye(n_base_points))

        self.kinematic_state = self._normalize_kinematic_state(
            kinematic_state,
            velocities,
        )
        self.kinematic_dim = self.kinematic_state.shape[0]
        self.velocities = self.kinematic_dim == 5
        self.shape_state = self._normalize_shape_state(shape_state, n_base_points)
        self.shape_dim = self.shape_state.shape[0]
        self.state = concatenate([self.kinematic_state, self.shape_state])

        self.kinematic_covariance = self._as_covariance_matrix(
            (
                0.1 * eye(self.kinematic_dim)
                if kinematic_covariance is None
                else kinematic_covariance
            ),
            self.kinematic_dim,
            "kinematic_covariance",
        )
        self.shape_covariance = self._as_covariance_matrix(
            self._k_uu if shape_covariance is None else shape_covariance,
            self.shape_dim,
            "shape_covariance",
        )
        if joint_covariance is None:
            self.covariance = linalg.block_diag(
                self.kinematic_covariance,
                self.shape_covariance,
            )
        else:
            self.covariance = self._as_covariance_matrix(
                joint_covariance,
                self.state.shape[0],
                "joint_covariance",
            )
        self._sync_state_views()

        self.dt = float(dt)
        self.sys_noise = self._as_covariance_matrix(
            (
                zeros((self.kinematic_dim, self.kinematic_dim))
                if sys_noise is None
                else sys_noise
            ),
            self.kinematic_dim,
            "sys_noise",
            require_positive_semidefinite=False,
        )
        self.acceleration_variance = float(acceleration_variance)
        self.extent_forgetting_rate = float(extent_forgetting_rate)
        self.reference_extent = self._normalize_shape_state(
            reference_extent,
            n_base_points,
        )
        self.radial_noise_variance = float(radial_noise_variance)
        self.measurement_noise = self._as_covariance_matrix(
            (
                zeros((self.measurement_dim, self.measurement_dim))
                if measurement_noise is None
                else measurement_noise
            ),
            self.measurement_dim,
            "measurement_noise",
            require_positive_semidefinite=False,
        )
        self.scale_mean = float(scale_mean)
        self.scale_variance = float(scale_variance)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.last_quadratic_form = None
        self.last_active_measurement_indices = None
        self.last_measurement_weights = None

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
        if matrix.ndim == 1:
            if matrix.shape[0] != dim:
                raise ValueError(f"{name} vector must have length {dim}")
            matrix = matrix * eye(dim)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        matrix = cls._symmetrize(matrix)
        if require_positive_semidefinite and not all(linalg.eigvalsh(matrix) >= -1e-12):
            raise ValueError(f"{name} must be positive semidefinite")
        return matrix

    @staticmethod
    def _normalize_kinematic_state(kinematic_state, velocities):
        if kinematic_state is None:
            return zeros(5 if velocities else 3)
        kinematic_state = array(kinematic_state)
        if kinematic_state.ndim != 1:
            raise ValueError("kinematic_state must be one-dimensional")
        if kinematic_state.shape[0] == 2:
            return concatenate([kinematic_state, zeros(3 if velocities else 1)])
        if kinematic_state.shape[0] == 3 and velocities:
            return concatenate([kinematic_state, zeros(2)])
        if kinematic_state.shape[0] in (3, 5):
            return kinematic_state
        raise ValueError("kinematic_state must have length 2, 3, or 5")

    @staticmethod
    def _normalize_shape_state(shape_state, shape_dim):
        if shape_state is None:
            return ones(shape_dim)
        shape_state = array(shape_state)
        if shape_state.shape != (shape_dim,):
            raise ValueError(f"shape_state must have shape ({shape_dim},)")
        return shape_state

    def _sync_state_views(self):
        self.kinematic_state = self.state[: self.kinematic_dim]
        self.shape_state = self.state[self.kinematic_dim :]
        self.kinematic_covariance = self.covariance[
            : self.kinematic_dim,
            : self.kinematic_dim,
        ]
        self.shape_covariance = self.covariance[
            self.kinematic_dim :,
            self.kinematic_dim :,
        ]
        self.m = self.kinematic_state
        self.p = self.shape_state
        self.C_m = self.kinematic_covariance
        self.C_p = self.shape_covariance

    def _periodic_kernel(self, angle_difference):
        exponent = 2.0 * sin(0.5 * abs(angle_difference)) ** 2 / self._kernel_width**2
        return self._kernel_variance * exp(-exponent)

    def _kernel_matrix(self, angles_1, angles_2):
        return array(
            [
                [self._periodic_kernel(angle_1 - angle_2) for angle_2 in angles_2]
                for angle_1 in angles_1
            ]
        )

    def _kernel_vector(self, phi):
        return array([[self._periodic_kernel(phi - phi_n) for phi_n in self.phi_pts]])

    def _basis_matrix(self, phi):
        if ndim(phi) == 0:
            return self._kernel_vector(phi) @ self._k_uu_inv
        return self._kernel_matrix(phi, self.phi_pts) @ self._k_uu_inv

    def _basis_derivative(self, phi):
        if ndim(phi) == 0:
            angles = array([phi])
        else:
            angles = phi
        return (
            array(
                [
                    [
                        -sin(angle - phi_n)
                        / self._kernel_width**2
                        * self._periodic_kernel(angle - phi_n)
                        for phi_n in self.phi_pts
                    ]
                    for angle in angles
                ]
            )
            @ self._k_uu_inv
        )

    def _residual_extent_covariance(self, phi):
        kernel_vector = self._kernel_vector(phi)
        residual_covariance = (
            self._periodic_kernel(0.0)
            - kernel_vector @ self._k_uu_inv @ kernel_vector.T
        )
        return residual_covariance[0, 0]

    def _transition_kinematics(self, kinematic_state, dt):
        transitioned = array(kinematic_state)
        orientation = kinematic_state[2]
        if self.velocities:
            speed = kinematic_state[3]
            turn_rate = kinematic_state[4]
            transitioned[0] = kinematic_state[0] + dt * speed * cos(orientation)
            transitioned[1] = kinematic_state[1] + dt * speed * sin(orientation)
            transitioned[2] = kinematic_state[2] + dt * turn_rate
        return transitioned

    def _transition_state(self, state, dt):
        kinematic_state = self._transition_kinematics(state[: self.kinematic_dim], dt)
        shape_decay = exp(-self.extent_forgetting_rate * dt)
        shape_state = self.reference_extent + shape_decay * (
            state[self.kinematic_dim :] - self.reference_extent
        )
        return concatenate([kinematic_state, shape_state])

    def _sigma_points(self, mean, covariance):
        dim = mean.shape[0]
        lambda_ = self.alpha**2 * (dim + self.kappa) - dim
        sigma_scale = dim + lambda_
        jitter = 1e-12 * eye(dim)
        chol = linalg.cholesky(sigma_scale * (covariance + jitter))
        sigma_points = [mean]
        for dim_index in range(dim):
            sigma_points.append(mean + chol[:, dim_index])
        for dim_index in range(dim):
            sigma_points.append(mean - chol[:, dim_index])
        mean_weights = zeros(2 * dim + 1)
        covariance_weights = zeros(2 * dim + 1)
        mean_weights[0] = lambda_ / sigma_scale
        covariance_weights[0] = mean_weights[0] + (1.0 - self.alpha**2 + self.beta)
        for weight_index in range(1, 2 * dim + 1):
            mean_weights[weight_index] = 0.5 / sigma_scale
            covariance_weights[weight_index] = 0.5 / sigma_scale
        return array(sigma_points), mean_weights, covariance_weights

    def _process_noise(self, dt):
        process_noise = linalg.block_diag(
            self.sys_noise,
            (1.0 - exp(-2.0 * self.extent_forgetting_rate * dt)) * self._k_uu,
        )
        if self.velocities and self.acceleration_variance > 0.0:
            orientation = self.kinematic_state[2]
            position_noise_direction = (
                0.5 * dt**2 * array([cos(orientation), sin(orientation)])
            )
            process_noise[0:2, 0:2] = process_noise[0:2, 0:2] + (
                self.acceleration_variance
                * outer(position_noise_direction, position_noise_direction)
            )
            process_noise[0:2, 3] = process_noise[0:2, 3] + (
                self.acceleration_variance * position_noise_direction * dt
            )
            process_noise[3, 0:2] = process_noise[0:2, 3]
            process_noise[3, 3] = process_noise[3, 3] + (
                self.acceleration_variance * dt**2
            )
        return self._symmetrize(process_noise)

    def predict(self, dt=None, sys_noise=None):
        if dt is None:
            dt = self.dt
        else:
            dt = float(dt)
        if sys_noise is None:
            process_noise = self._process_noise(dt)
        else:
            process_noise = linalg.block_diag(
                self._as_covariance_matrix(
                    sys_noise,
                    self.kinematic_dim,
                    "sys_noise",
                    require_positive_semidefinite=False,
                ),
                (1.0 - exp(-2.0 * self.extent_forgetting_rate * dt)) * self._k_uu,
            )

        sigma_points, mean_weights, covariance_weights = self._sigma_points(
            self.state,
            self.covariance,
        )
        propagated_sigma_points = array(
            [self._transition_state(sigma_point, dt) for sigma_point in sigma_points]
        )
        predicted_state = zeros(self.state.shape[0])
        for point_index in range(propagated_sigma_points.shape[0]):
            predicted_state = (
                predicted_state
                + mean_weights[point_index] * propagated_sigma_points[point_index]
            )
        predicted_covariance = zeros(self.covariance.shape)
        for point_index in range(propagated_sigma_points.shape[0]):
            diff = propagated_sigma_points[point_index] - predicted_state
            predicted_covariance = predicted_covariance + (
                covariance_weights[point_index] * outer(diff, diff)
            )
        self.state = predicted_state
        self.covariance = self._symmetrize(predicted_covariance + process_noise)
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
            "measurements must have shape (2, n_measurements) or (n_measurements, 2)"
        )

    def _measurement_model_terms(self, measurement, measurement_noise):
        position = self.kinematic_state[:2]
        orientation = self.kinematic_state[2]
        delta = measurement - position
        delta_norm = linalg.norm(delta)
        if float(delta_norm) <= 1e-12:
            unit_direction = array([cos(orientation), sin(orientation)])
            delta_norm = 1.0
        else:
            unit_direction = delta / delta_norm

        theta = arctan2(unit_direction[1], unit_direction[0]) - orientation
        basis_row = self._basis_matrix(theta)[0]
        basis_derivative_row = self._basis_derivative(theta)[0]
        radius = basis_row @ self.shape_state
        radius_derivative = basis_derivative_row @ self.shape_state
        predicted_measurement = position + self.scale_mean * unit_direction * radius

        center_direction_jacobian = (
            outer(unit_direction, unit_direction) - eye(2)
        ) / delta_norm
        theta_center_jacobian = (
            array([unit_direction[1], -unit_direction[0]]) / delta_norm
        )
        measurement_jacobian = zeros((self.measurement_dim, self.state.shape[0]))
        measurement_jacobian[:, :2] = eye(2) + self.scale_mean * (
            center_direction_jacobian * radius
            + outer(unit_direction, theta_center_jacobian) * radius_derivative
        )
        measurement_jacobian[:, 2] = (
            -self.scale_mean * unit_direction * radius_derivative
        )
        measurement_jacobian[:, self.kinematic_dim :] = self.scale_mean * outer(
            unit_direction,
            basis_row,
        )

        radial_variance = self.radial_noise_variance + self._residual_extent_covariance(
            theta
        )
        if float(radial_variance) < 0.0:
            radial_variance = 0.0
        noise_covariance = measurement_noise + radial_variance * outer(
            unit_direction,
            unit_direction,
        )
        if self.scale_variance > 0.0:
            radial_second_moment = (
                basis_row @ self.shape_covariance @ basis_row.T
                + self._residual_extent_covariance(theta)
                + radius**2
            )
            noise_covariance = noise_covariance + (
                self.scale_variance
                * radial_second_moment
                * outer(unit_direction, unit_direction)
            )
        return measurement_jacobian, predicted_measurement, noise_covariance

    def _normalize_measurement_weights(self, measurement_weights, n_measurements):
        if measurement_weights is None:
            return ones(n_measurements)

        weights = array(measurement_weights)
        if weights.ndim == 0:
            weights = ones(n_measurements) * float(weights)
        else:
            weights = reshape(weights, (-1,))
            if weights.shape[0] != n_measurements:
                raise ValueError(
                    "measurement_weights must be scalar or have one entry per measurement"
                )
        if not bool(all(isfinite(weights))):
            raise ValueError("measurement_weights must be finite")
        if not bool(all(weights >= 0.0)):
            raise ValueError("measurement_weights must be non-negative")
        return weights

    def _normalize_active_measurement_mask(self, active_measurement_mask, n_measurements):
        if active_measurement_mask is None:
            return [True] * n_measurements

        mask = array(active_measurement_mask)
        if mask.ndim == 0:
            return [bool(mask)] * n_measurements
        mask = reshape(mask, (-1,))
        if mask.shape[0] != n_measurements:
            raise ValueError(
                "active_measurement_mask must be scalar or have one entry per measurement"
            )
        return [bool(mask[index]) for index in range(n_measurements)]

    def _stack_measurement_terms(
        self,
        measurements,
        measurement_noise,
        measurement_weights=None,
        active_measurement_mask=None,
    ):
        weights = self._normalize_measurement_weights(
            measurement_weights,
            measurements.shape[0],
        )
        active_mask = self._normalize_active_measurement_mask(
            active_measurement_mask,
            measurements.shape[0],
        )
        measurement_jacobians = []
        predicted_measurements = []
        noise_covariances = []
        active_indices = []
        for measurement_index, measurement in enumerate(measurements):
            weight = float(weights[measurement_index])
            if not active_mask[measurement_index] or weight <= 0.0:
                continue
            measurement_jacobian, predicted_measurement, noise_covariance = (
                self._measurement_model_terms(measurement, measurement_noise)
            )
            active_indices.append(measurement_index)
            measurement_jacobians.append(measurement_jacobian)
            predicted_measurements.append(predicted_measurement)
            noise_covariances.append(noise_covariance / weight)
        self.last_measurement_weights = weights
        self.last_active_measurement_indices = active_indices
        if not active_indices:
            return None, None, None, active_indices
        return (
            vstack(measurement_jacobians),
            concatenate(predicted_measurements),
            linalg.block_diag(*noise_covariances),
            active_indices,
        )

    def update(
        self,
        measurements,
        R=None,
        s_hat=None,
        sigma_squared_s=None,
        measurement_weights=None,
        active_measurement_mask=None,
    ):
        """Update the tracker with optional per-measurement reliabilities.

        ``measurement_weights`` scales each measurement covariance block as
        ``R_i / weight_i``. Zero-weight or masked measurements are skipped.
        ``active_measurement_mask`` can be used to explicitly disable cluttered,
        occluded, or otherwise unsupported measurements.
        """
        if s_hat is not None:
            self.scale_mean = float(s_hat)
        if sigma_squared_s is not None:
            self.scale_variance = float(sigma_squared_s)
        measurements = self._normalize_measurements(measurements)
        measurement_noise = self.measurement_noise
        if R is not None:
            measurement_noise = self._as_covariance_matrix(
                R,
                self.measurement_dim,
                "R",
                require_positive_semidefinite=False,
            )

        measurement_jacobian, predicted_measurements, noise_covariance, active_indices = (
            self._stack_measurement_terms(
                measurements,
                measurement_noise,
                measurement_weights=measurement_weights,
                active_measurement_mask=active_measurement_mask,
            )
        )
        if not active_indices:
            self.last_quadratic_form = None
            return

        residual = concatenate([measurements[index] for index in active_indices])
        residual = residual - predicted_measurements
        covariance_measurement = self._symmetrize(
            measurement_jacobian @ self.covariance @ measurement_jacobian.T
            + noise_covariance
        )
        cross_covariance = self.covariance @ measurement_jacobian.T
        gain = linalg.solve(covariance_measurement.T, cross_covariance.T).T
        self.state = self.state + gain @ residual
        self.covariance = self._symmetrize(
            self.covariance - gain @ covariance_measurement @ gain.T
        )
        self._sync_state_views()
        solved_residual = linalg.solve(covariance_measurement, residual)
        self.last_quadratic_form = residual @ solved_residual

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def get_point_estimate(self):
        return self.state

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_shape(self):
        return self.shape_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        return self.shape_state.flatten() if flatten_matrix else self.shape_state

    def get_extents_on_grid(self, n: int = 100, angles=None, body_frame=False):
        if angles is None:
            angles = linspace(0.0, 2 * pi, n, endpoint=False)
        if body_frame:
            body_angles = angles
        else:
            body_angles = angles - self.kinematic_state[2]
        return self._basis_matrix(body_angles) @ self.shape_state

    def get_contour_points(self, n: int = 100, scaling_factor=1.0):
        angles = linspace(0.0, 2 * pi, n, endpoint=False)
        extents = scaling_factor * self.get_extents_on_grid(n, angles=angles)
        coords = pol2cart(angles, extents) + reshape(self.kinematic_state[:2], (-1, 1))
        return coords.T

    def get_bounding_box(self, n: int = 100):
        contour_points = self.get_contour_points(n)
        min_xy = array([min(contour_points[:, 0]), min(contour_points[:, 1])])
        max_xy = array([max(contour_points[:, 0]), max(contour_points[:, 1])])
        return {
            "center_xy": 0.5 * (min_xy + max_xy),
            "dimension": max_xy - min_xy,
            "orientation": self.kinematic_state[2],
        }


class DecorrelatedSCGPTracker(FullSCGPTracker):
    """SCGP tracker variant with zeroed kinematic-shape cross covariance."""

    def _zero_cross_covariance(self):
        self.covariance[: self.kinematic_dim, self.kinematic_dim :] = 0.0
        self.covariance[self.kinematic_dim :, : self.kinematic_dim] = 0.0
        self._sync_state_views()

    def predict(self, *args, **kwargs):
        self._zero_cross_covariance()
        super().predict(*args, **kwargs)
        self._zero_cross_covariance()

    def update(self, *args, **kwargs):
        self._zero_cross_covariance()
        super().update(*args, **kwargs)
        self._zero_cross_covariance()


SCGPTracker = FullSCGPTracker
ScGpTracker = FullSCGPTracker
DecorrelatedScGpTracker = DecorrelatedSCGPTracker
