# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    concatenate,
    cos,
    exp,
    eye,
    linalg,
    linspace,
    mean,
    pi,
    sin,
)
from pyrecest.utils.plotting import plot_ellipsoid

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker
import warnings


class RandomMatrixTracker(AbstractExtendedObjectTracker):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        kinematic_state,
        covariance,
        extent,
        kinematic_state_to_pos_matrix=None,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extent=False,
        log_posterior_extent=False,
    ):
        super().__init__(
            log_prior_estimates,
            log_posterior_estimates,
            log_prior_extent,
            log_posterior_extent,
        )
        self.kinematic_state = kinematic_state  # Initial kinematic state
        self.covariance = covariance  # Initial state covariance matrix
        self.extent = extent  # Initial extent matrix
        self.alpha = 0
        self.kinematic_state_to_pos_matrix = kinematic_state_to_pos_matrix

    def get_point_estimate(self):
        # Combines the kinematic state and flattened extent matrix into one vector
        return concatenate([self.kinematic_state, self.extent.flatten()])

    def get_point_estimate_kinematics(self):
        # Returns just the kinematic state
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        # Returns just the extent matrix
        if flatten_matrix:
            return self.extent.flatten()
        return self.extent

    def predict_linear(self, sys_noise_cov, system_matrix, delta_t:float|int = 1.0, tau:float|int = 1.0):
        F = system_matrix
        x_rows = self.kinematic_state.shape[0]
        y_rows = x_rows // 2

        if Cw.shape in ((), (1,)):
            Cw = Cw * eye(x_rows)

        self.kinematic_state = F @ self.kinematic_state
        self.covariance = F @ self.covariance @ F.T + sys_noise_cov

        self.alpha = y_rows + exp(-delta_t / tau) * (self.alpha - y_rows)

    def update_using_position_measurements(self, measurement, meas_noise_cov):
        assert self.kinematic_state_to_pos_matrix is not None
        self.update_linear(measurement, self.kinematic_state_to_pos_matrix, meas_noise_cov)
    
    # pylint: disable=too-many-locals
    def update_linear(self, measurements, meas_mat, meas_noise_cov):
        if self.kinematic_state_to_pos_matrix is None:
            # Usually, the measurement matrix is mapping the kinematic state to the position.
            self.kinematic_state_to_pos_matrix = meas_mat
        if np.size(measurements) == 0:
            warnings.warn("No measurements given, skipping update step.")
            return

        Cv = meas_noise_cov
        ys = measurements
        H = meas_mat

        y_rows, y_cols = ys.shape
        if y_cols < y_rows + 1:
            raise ValueError("Too few measurements.")

        y_ = mean(ys, axis=1, keepdims=True)
        ys_demean = ys - y_
        Y_ = ys_demean @ ys_demean.T

        Hx = H @ self.kinematic_state

        Y = self.extent + Cv
        S = H @ self.covariance @ H.T + Y / y_cols
        K = self.covariance @ linalg.solve(S, H).T
        self.kinematic_state = self.kinematic_state + K @ (y_.flatten() - Hx)
        self.covariance = self.covariance - K @ S @ K.T

        Xsqrt = linalg.cholesky(self.extent)
        Ssqrt = linalg.cholesky(S)
        Ysqrt = linalg.cholesky(Y)

        Nsqrt = Xsqrt * linalg.inv(Ssqrt) @ (y_ - Hx)
        N = Nsqrt @ Nsqrt.T
        XYsqrt = Xsqrt * linalg.inv(Ysqrt)

        self.extent = (self.alpha * self.extent + N + XYsqrt @ Y_ @ XYsqrt.T) / (
            self.alpha + y_cols
        )

        self.alpha = self.alpha + y_cols

    def plot_point_estimate(self, scaling_factor=1, color=(0, 0.4470, 0.7410)):
        if self.kinematic_state_to_pos_matrix is None:
            raise ValueError(
                """No kinematic_state_to_pos_matrix
                             matrix was set, so it is unclear what
                             the individual components of the kinematic
                             state are (position, velocity, etc.).
                             Please set it directly or perform an update step
                             before plotting."""
            )
        position_estimate = self.kinematic_state_to_pos_matrix @ self.kinematic_state
        plot_ellipsoid(position_estimate, self.extent, scaling_factor, color)

    def get_contour_points(self, n):
        assert self.kinematic_state.shape == (2,)
        xs = linspace(0, 2 * pi, n)
        contour_points = self.kinematic_state[:, None] + self.extent @ array(
            [cos(xs), sin(xs)]
        )
        return contour_points.T
