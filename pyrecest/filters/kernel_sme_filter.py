import filterpy
import numpy as np
import scipy
from pyrecest.distributions import GaussianDistribution
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist
from scipy.stats import chi2, multivariate_normal

from .abstract_multitarget_tracker import AbstractMultitargetTracker


class KernelSMEFilter(AbstractMultitargetTracker):
    def __init__(
        self,
        initial_priors: list[GaussianDistribution] | GaussianDistribution | None = None,
        log_estimates: bool = True,
    ):
        AbstractMultitargetTracker.__init__(
            self,
            log_prior_estimates=log_estimates,
            log_posterior_estimates=log_estimates,
        )

        if initial_priors is None:
            self.x = None
            self.C = None
            self.n_targets = 0
        else:
            self.n_targets = np.size(initial_priors)
            self.filter_state = initial_priors  # type: ignore

    @property
    def filter_state(self) -> GaussianDistribution:
        return GaussianDistribution(self.x, self.C)

    @filter_state.setter
    def filter_state(self, value: list[GaussianDistribution] | GaussianDistribution):
        if isinstance(value, GaussianDistribution):
            value = [value]
        self.n_targets = len(value)
        x_list = [prior.mu for prior in value]
        C_list = [prior.C for prior in value]
        self.x = np.hstack(x_list)
        self.C = scipy.linalg.block_diag(*C_list)
        if self.log_prior_estimates:
            self.store_prior_estimates()

    def get_point_estimate(self, flatten_vector: bool = False):
        est = self.x
        if not flatten_vector:
            est = np.reshape(est, (-1, self.n_targets), order="F")
        return est

    def get_number_of_targets(self) -> int:
        return self.n_targets

    def predict_linear(
        self,
        system_matrix: np.ndarray,
        sys_noise: np.ndarray,
        inputs: np.ndarray | None = None,
    ):
        assert (
            inputs is None
        ), "Inputs are currently not supported for the Kernel SME filter"
        if isinstance(sys_noise, GaussianDistribution):
            sys_noise_cov = sys_noise.C
        else:
            sys_noise_cov = sys_noise
        # Reshape into matrix, multiply with system matrix, reshape into vector
        self.x = (system_matrix @ self.get_point_estimate(flatten_vector=False)).ravel(
            order="F"
        )
        sys_mat_list = [system_matrix] * self.n_targets
        sys_noise_cov_list = [sys_noise_cov] * self.n_targets
        self.C = block_diag(*sys_mat_list) @ self.C @ block_diag(
            *sys_mat_list
        ).T + block_diag(*sys_noise_cov_list)
        if self.log_prior_estimates:
            self.store_prior_estimates()

    @property
    def dim(self) -> int:
        return np.size(self.x) // self.n_targets

    # pylint: disable=too-many-arguments,too-many-locals
    def update_linear(
        self,
        measurements,
        measurement_matrix,
        cov_mat_meas,
        false_alarm_rate=0,
        clutter_cov=None,
        lambda_multimeas=1,
        enable_gating=False,
        gating_threshold=None,
    ):
        """
        Update the filter with new measurements using the linear measurement model.

        Args:
            measurements (numpy.ndarray): Array of shape `(dim_meas, n_meas)` containing the measurements.
            measurementMatrix (numpy.ndarray): Array of shape `(dim_meas, dim_state)` containing the measurement matrix.
            covMatMeas (numpy.ndarray): Array of shape `(dim_meas, dim_meas)` containing the measurement noise covariance matrix.
            falseAlarmRate (float): Scalar representing the false alarm rate. Default is 0, which means that clutter is disabled.
            clutterCov (numpy.ndarray): Array of shape `(dim_meas, dim_meas)` containing the clutter covariance matrix. Default is None, which means that clutter is disabled.
            lambdaMultimeas (float): Scalar representing the scaling factor for multimeasurements. Default is 1, which means that multimeasurements are not used.
            enableGating (bool): If True, gating will be used to remove unlikely measurements before the update. Default is False.
            gatingThreshold (float): Scalar representing the gating threshold. Default is None, which means that the threshold is set to chi2inv(0.99, dim_meas).

        Raises:
            AssertionError: If `enableGating=True` and `gatingThreshold` is None.

        """
        assert (
            not gating_threshold or enable_gating
        ), "Changed gating threshold without enabling gating, this does not make sense."
        if gating_threshold is None:
            gating_threshold = chi2.ppf(0.99, len(measurements))
        n_meas = measurements.shape[1]
        kernel_width = np.mean(np.diag(cov_mat_meas)) ** 2

        if enable_gating:
            dists = np.full((self.n_targets, n_meas), np.nan)
            state_dim = np.size(self.x) // self.n_targets
            n_blocks = self.n_targets
            C_blocks_list = [
                self.C[
                    i * state_dim : (i + 1) * state_dim,  # noqa: E203
                    i * state_dim : (i + 1) * state_dim,  # noqa: E203
                ]
                for i in range(n_blocks)
            ]

            for i in range(self.n_targets):
                dists[i, :] = cdist(  # noqa: E203
                    (measurement_matrix @ self.get_point_estimate()[:, i]).T[
                        np.newaxis
                    ],
                    measurements.T,
                    metric="mahalanobis",
                    VI=(
                        measurement_matrix @ C_blocks_list[i] @ measurement_matrix.T
                        + cov_mat_meas
                    ),
                )
            dists = dists**2  # Squared Mahalanobis distances are usually considered
            # Filter out all that are incompatible with all tracks
            measurements = measurements[:, np.any(dists < gating_threshold, axis=0)]

        test_points = KernelSMEFilter.gen_test_points(measurements, kernel_width)
        pseudo_meas = KernelSMEFilter.calc_pseudo_meas(
            test_points, measurements, kernel_width
        )

        mu_s, sigma_s, sigma_xs = KernelSMEFilter.calc_moments(
            self.x,
            self.C,
            measurement_matrix,
            cov_mat_meas,
            test_points,
            kernel_width,
            self.n_targets,
            false_alarm_rate,
            clutter_cov,
            lambda_multimeas,
        )

        if np.linalg.cond(sigma_s) > 1e10:
            sigma_s = sigma_s + 1e-5 * np.eye(sigma_s.shape[0])

        x_posterior = self.x + sigma_xs @ np.linalg.solve(sigma_s, pseudo_meas - mu_s)
        self.C = self.C - sigma_xs @ np.linalg.solve(sigma_s, sigma_xs.T)
        self.x = x_posterior

        if self.log_posterior_estimates:
            self.store_posterior_estimates()

    @staticmethod
    def gen_test_points(measurements, kernel_width):
        meas_dim = measurements.shape[0]
        n_meas = measurements.shape[1]

        # Use high level function from filterpy to generate sigma points
        # Reorder samples to be consistent with the Matlab implementation
        all_sample_points_list = [
            filterpy.kalman.JulierSigmaPoints(meas_dim).sigma_points(
                measurements[:, i], np.sqrt(kernel_width) * np.eye(meas_dim)
            )[
                [0, 3, 1, 4, 2], :  # noqa: E203
            ]
            for i in range(n_meas)
        ]

        return np.vstack(all_sample_points_list).T

    @staticmethod
    def calc_pseudo_meas(testPoints, measurements, kernel_width):
        nMeas = measurements.shape[1]
        measDim = measurements.shape[0]
        nTestPoints = testPoints.shape[1]
        pseudoMeas = np.zeros(nTestPoints)
        for i in range(nTestPoints):
            a_i = testPoints[:, i]
            for j in range(nMeas):
                pseudoMeas[i] += multivariate_normal.pdf(
                    a_i, mean=measurements[:, j], cov=kernel_width * np.eye(measDim)
                )
        return pseudoMeas

    # pylint: disable=too-many-arguments,too-many-locals
    @staticmethod
    def calc_moments(
        x_prior,
        C_prior,
        measurement_matrix,
        covMatMeas,
        testPoints,
        kernel_width,
        n_targets,
        falseAlarmRate=0,
        clutterCov=None,
        lambdaMultimeas=1,
    ):
        if clutterCov is None:
            clutterCov = np.zeros_like(covMatMeas)
        if isinstance(lambdaMultimeas, (float, int)):
            lambdaMultimeas = lambdaMultimeas * np.ones(n_targets)
        n_testpoints = testPoints.shape[1]
        state_dim = np.size(x_prior) // n_targets

        P = np.empty((n_testpoints, n_targets))
        P[:] = np.nan

        n_blocks = np.size(x_prior) // state_dim
        C_blocks_list = [
            C_prior[
                i * state_dim : (i + 1) * state_dim,  # noqa: E203
                i * state_dim : (i + 1) * state_dim,  # noqa: E203
            ]
            for i in range(n_blocks)
        ]
        x_prior_mat = np.reshape(x_prior, (state_dim, n_targets), order="F")

        for k in range(n_targets):
            sTemp = (
                measurement_matrix @ C_blocks_list[k] @ measurement_matrix.T
                + covMatMeas
                + kernel_width * np.eye(covMatMeas.shape[0])
            )
            muTemp = measurement_matrix @ x_prior_mat[:, k]
            for i in range(n_testpoints):
                P[i, k] = multivariate_normal.pdf(
                    testPoints[:, i], mean=muTemp, cov=sTemp
                ) + falseAlarmRate * multivariate_normal.pdf(
                    testPoints[:, i],
                    mean=np.zeros(measurement_matrix.shape[0]),
                    cov=clutterCov + kernel_width * np.eye(measurement_matrix.shape[0]),
                )

        Pij = np.zeros((n_testpoints, n_testpoints, n_targets))
        for k in range(n_targets):
            sTemp = (
                measurement_matrix @ C_blocks_list[k] @ measurement_matrix.T
                + covMatMeas
                + 0.5 * kernel_width * np.eye(covMatMeas.shape[0])
            )
            muTemp = measurement_matrix @ x_prior_mat[:, k]
            for i in range(n_testpoints):
                for j in range(n_testpoints):
                    Pij[i, j, k] = multivariate_normal.pdf(
                        0.5 * (testPoints[:, i] + testPoints[:, j]),
                        mean=muTemp,
                        cov=0.5 * (sTemp + sTemp).T,
                    )

        mu_s = np.sum(P, axis=1)

        sigma_s = np.zeros((n_testpoints, n_testpoints))
        for i in range(n_testpoints):
            for j in range(i, n_testpoints):
                M = np.sum(P[j, :])  # noqa: E203
                P_temp = multivariate_normal.pdf(
                    testPoints[:, i],
                    mean=testPoints[:, j],
                    cov=2 * kernel_width * np.eye(covMatMeas.shape[0]),
                )
                sigma_s[i, j] = (
                    np.sum(
                        lambdaMultimeas
                        * (
                            P_temp * Pij[i, j, :]
                            + P[i, :] * (M - P[j, :])  # noqa: E203
                        )
                    )
                    + sigma_s[i, j]
                    + falseAlarmRate**2
                    * multivariate_normal.pdf(
                        testPoints[:, i],
                        mean=np.zeros(measurement_matrix.shape[0]),
                        cov=clutterCov + kernel_width * np.eye(covMatMeas.shape[0]),
                    )
                    * multivariate_normal.pdf(
                        testPoints[:, j],
                        mean=np.zeros(measurement_matrix.shape[0]),
                        cov=clutterCov + kernel_width * np.eye(covMatMeas.shape[0]),
                    )
                    + falseAlarmRate
                    * multivariate_normal.pdf(
                        testPoints[:, i],
                        mean=testPoints[:, j],
                        cov=2 * kernel_width * np.eye(covMatMeas.shape[0]),
                    )
                    * multivariate_normal.pdf(
                        0.5 * (testPoints[:, i] + testPoints[:, j]),
                        mean=np.zeros(measurement_matrix.shape[0]),
                        cov=clutterCov + kernel_width * np.eye(covMatMeas.shape[0]),
                    )
                    - mu_s[i] * mu_s[j]
                    - falseAlarmRate**2
                    * multivariate_normal.pdf(
                        testPoints[:, i],
                        mean=np.zeros(measurement_matrix.shape[0]),
                        cov=clutterCov + kernel_width * np.eye(covMatMeas.shape[0]),
                    )
                    * multivariate_normal.pdf(
                        testPoints[:, j],
                        mean=np.zeros(measurement_matrix.shape[0]),
                        cov=clutterCov + kernel_width * np.eye(covMatMeas.shape[0]),
                    )
                )
        sigma_s += np.triu(sigma_s, 1).T

        sigma_xs = np.zeros((x_prior.shape[0], n_testpoints))
        C_k_full_columns = np.hsplit(C_prior, n_targets)
        K = [
            C_k_full_columns[k]
            @ measurement_matrix.T
            @ np.linalg.inv(
                measurement_matrix @ C_blocks_list[k] @ measurement_matrix.T
                + covMatMeas
                + kernel_width * np.eye(covMatMeas.shape[0])
            )
            for k in range(n_targets)
        ]
        for i in range(n_testpoints):
            for k in range(n_targets):
                sigma_xs[:, i] += (
                    lambdaMultimeas[k]
                    * P[i, k]
                    * (
                        x_prior
                        + K[k]
                        @ (testPoints[:, i] - measurement_matrix @ x_prior_mat[:, k])
                    )
                )
            sigma_xs[:, i] -= mu_s[i] * x_prior
        return mu_s, sigma_s, sigma_xs
