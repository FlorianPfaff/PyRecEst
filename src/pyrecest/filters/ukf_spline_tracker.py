# pylint: disable=duplicate-code,no-member,no-name-in-module,too-many-instance-attributes
from pyrecest.backend import array, concatenate, eye, linalg, outer, zeros

from .ekf_spline_tracker import EKFSplineTracker


class UKFSplineTracker(EKFSplineTracker):
    """UKF tracker for a 2-D extended object with a closed spline extent.

    The state is ``[x, y, orientation, speed, turn_rate, scale_x, scale_y]``.
    The spline contour handling is shared with :class:`EKFSplineTracker`, while
    prediction and correction use an unscented transform instead of EKF
    Jacobians.
    """

    minimum_scale = 1e-9

    # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        alpha=1.0,
        beta=2.0,
        kappa=2.0,
        finite_difference_step=1e-5,
        closest_point_grid_size=11,
        closest_point_iterations=8,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        super().__init__(
            control_points=control_points,
            kinematic_state=kinematic_state,
            scale_state=scale_state,
            covariance=covariance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=dt,
            acceleration_variance=acceleration_variance,
            turn_rate_variance=turn_rate_variance,
            scale_process_noise=scale_process_noise,
            scale_correction=scale_correction,
            orientation_correction=orientation_correction,
            finite_difference_step=finite_difference_step,
            closest_point_grid_size=closest_point_grid_size,
            closest_point_iterations=closest_point_iterations,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.state_dim + self._sigma_point_lambda(self.state_dim) <= 0.0:
            raise ValueError("alpha and kappa must yield a positive sigma spread")

    def _sigma_point_lambda(self, dim):
        return self.alpha**2 * (dim + self.kappa) - dim

    def _sigma_point_weights(self, dim):
        lambda_value = self._sigma_point_lambda(dim)
        spread = dim + lambda_value
        mean_weights = zeros(2 * dim + 1)
        covariance_weights = zeros(2 * dim + 1)
        mean_weights[0] = lambda_value / spread
        covariance_weights[0] = mean_weights[0] + 1.0 - self.alpha**2 + self.beta
        for index in range(1, 2 * dim + 1):
            mean_weights[index] = 0.5 / spread
            covariance_weights[index] = 0.5 / spread
        return mean_weights, covariance_weights, spread

    @staticmethod
    def _cholesky_with_jitter(covariance):
        jitter = 0.0
        for _ in range(6):
            try:
                return linalg.cholesky(covariance + jitter * eye(covariance.shape[0]))
            except Exception:  # pylint: disable=broad-exception-caught
                jitter = 1e-12 if jitter == 0.0 else 10.0 * jitter
        return linalg.cholesky(covariance + jitter * eye(covariance.shape[0]))

    def _sigma_points(self, mean, covariance):
        dim = mean.shape[0]
        mean_weights, covariance_weights, spread = self._sigma_point_weights(dim)
        covariance = self._symmetrize(covariance)
        cholesky_factor = self._cholesky_with_jitter(spread * covariance)
        sigma_points = [mean]
        for dim_index in range(dim):
            direction = cholesky_factor[:, dim_index]
            sigma_points.append(mean + direction)
            sigma_points.append(mean - direction)
        return array(sigma_points), mean_weights, covariance_weights

    @staticmethod
    def _weighted_mean(values, mean_weights):
        mean = zeros(values.shape[1])
        for index in range(values.shape[0]):
            mean = mean + mean_weights[index] * values[index]
        return mean

    @staticmethod
    def _weighted_covariance(left_diffs, right_diffs, covariance_weights):
        covariance = zeros((left_diffs.shape[1], right_diffs.shape[1]))
        for index in range(left_diffs.shape[0]):
            covariance = covariance + covariance_weights[index] * outer(
                left_diffs[index],
                right_diffs[index],
            )
        return covariance

    def _enforce_positive_scales(self, sigma_points):
        adjusted_sigma_points = array(sigma_points)
        for point_index in range(adjusted_sigma_points.shape[0]):
            for state_index in range(self.state_dim - 2, self.state_dim):
                if float(adjusted_sigma_points[point_index, state_index]) <= self.minimum_scale:
                    adjusted_sigma_points[point_index, state_index] = self.minimum_scale
        return adjusted_sigma_points

    def _enforce_state_scale_bounds(self):
        for state_index in range(self.state_dim - 2, self.state_dim):
            if float(self.state[state_index]) <= self.minimum_scale:
                self.state[state_index] = self.minimum_scale

    def _measurement_sigma_points(self, sigma_points):
        measurement_sigma_points = array(sigma_points)
        if not self.orientation_correction:
            measurement_sigma_points[:, 2] = self.state[2]
        if not self.scale_correction:
            measurement_sigma_points[:, -2:] = self.state[-2:]
        return self._enforce_positive_scales(measurement_sigma_points)

    def predict(self, dt=None, process_noise=None):
        if dt is None:
            dt = self.dt
        else:
            dt = float(dt)
        if process_noise is None:
            process_noise = self._process_noise(dt)
        else:
            process_noise = self._as_covariance_matrix(
                process_noise,
                self.state_dim,
                "process_noise",
                require_positive_semidefinite=False,
            )

        sigma_points, mean_weights, covariance_weights = self._sigma_points(
            self.state,
            self.covariance,
        )
        propagated_sigma_points = array(
            [self._transition_state(point, dt) for point in sigma_points]
        )
        predicted_state = self._weighted_mean(propagated_sigma_points, mean_weights)
        state_diffs = propagated_sigma_points - predicted_state
        self.state = predicted_state
        self._enforce_state_scale_bounds()
        self.covariance = self._symmetrize(
            self._weighted_covariance(
                state_diffs,
                state_diffs,
                covariance_weights,
            )
            + process_noise
        )
        self._sync_state_views()

        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def update(self, measurements, R=None):
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

        sigma_points, mean_weights, covariance_weights = self._sigma_points(
            self.state,
            self.covariance,
        )
        measurement_sigma_points = self._measurement_sigma_points(sigma_points)
        predicted_measurement_points = array(
            [
                self._predict_measurements_from_state(point, measurements)
                for point in measurement_sigma_points
            ]
        )
        predicted_measurement = self._weighted_mean(
            predicted_measurement_points,
            mean_weights,
        )
        state_diffs = sigma_points - self.state
        measurement_diffs = predicted_measurement_points - predicted_measurement

        block_measurement_noise = linalg.block_diag(
            *[measurement_noise for _ in range(measurements.shape[0])]
        )
        innovation_covariance = self._symmetrize(
            self._weighted_covariance(
                measurement_diffs,
                measurement_diffs,
                covariance_weights,
            )
            + block_measurement_noise
        )
        cross_covariance = self._weighted_covariance(
            state_diffs,
            measurement_diffs,
            covariance_weights,
        )
        gain = linalg.solve(innovation_covariance.T, cross_covariance.T).T

        stacked_measurements = concatenate(list(measurements))
        residual = stacked_measurements - predicted_measurement
        self.state = self.state + gain @ residual
        self._enforce_state_scale_bounds()
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


UkfSplineTracker = UKFSplineTracker
