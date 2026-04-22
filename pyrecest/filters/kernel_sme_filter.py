import warnings

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    any,
    array,
    diag,
    dot,
    einsum,
    empty,
    eye,
    full,
    hsplit,
    hstack,
    linalg,
    mean,
    ones,
    reshape,
    sqrt,
    stack,
    sum,
    triu_indices,
    vstack,
    zeros,
    zeros_like,
)
from pyrecest.distributions import GaussianDistribution

from ._ukf import JulierSigmaPoints
from .abstract_multitarget_tracker import AbstractMultitargetTracker


class KernelSMEFilter(AbstractMultitargetTracker):
    """
    Using the Kernel Symmetric Measurement Equation (SME) approach for multitarget tracking.
    Measurements are assumed to be (meas_dim, n_meas) arrays because this simplifies algebraic
    operations in the implementation.
    """

    def __init__(
        self,
        initial_priors: list[GaussianDistribution] | None = None,
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
            self.n_targets = len(initial_priors)
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
        self.x = hstack(x_list)
        self.C = linalg.block_diag(*C_list)
        if self.log_prior_estimates:
            self.store_prior_estimates()

    def get_point_estimate(self, flatten_vector: bool = False):
        est = self.x
        if not flatten_vector:
            # Pytorch does not support "order=F" in reshape, so we do it manually
            est = reshape(est, (self.n_targets, -1)).T
        return est

    def get_number_of_targets(self) -> int:
        return self.n_targets

    def predict_linear(
        self,
        system_matrix,
        sys_noise,
        inputs=None,
    ):
        assert (
            inputs is None
        ), "Inputs are currently not supported for the Kernel SME filter"
        if isinstance(sys_noise, GaussianDistribution):
            sys_noise_cov = sys_noise.C
        else:
            sys_noise_cov = sys_noise
        # Reshape into matrix, multiply with system matrix, reshape into vector
        # Pytorch does not support "order=F" in reshape, so we do it manually
        self.x = (
            system_matrix @ self.get_point_estimate(flatten_vector=False)
        ).T.reshape((-1,))

        sys_mat_list = [system_matrix] * self.n_targets
        sys_noise_cov_list = [sys_noise_cov] * self.n_targets
        self.C = linalg.block_diag(*sys_mat_list) @ self.C @ linalg.block_diag(
            *sys_mat_list
        ).T + linalg.block_diag(*sys_noise_cov_list)
        if self.log_prior_estimates:
            self.store_prior_estimates()

    @property
    def dim(self) -> int:
        if self.x is None:
            warnings.warn("Filter state is not initialized yet, output 0 as dimension.")
            return 0
        return self.x.shape[0] // self.n_targets

    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
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
        from scipy.stats import (  # Just a simple calculation, do not need to use backend
            chi2,
        )

        assert (
            not gating_threshold or enable_gating
        ), "Changed gating threshold without enabling gating, this does not make sense."
        if gating_threshold is None:
            gating_threshold = chi2.ppf(0.99, len(measurements))
        n_meas = measurements.shape[1]
        kernel_width = mean(diag(cov_mat_meas)) ** 2

        if enable_gating:
            dists = full((self.n_targets, n_meas), float("nan"))
            state_dim = self.x.shape[0] // self.n_targets
            n_blocks = self.n_targets
            C_blocks_list = [
                self.C[
                    i * state_dim : (i + 1) * state_dim,  # noqa: E203
                    i * state_dim : (i + 1) * state_dim,  # noqa: E203
                ]
                for i in range(n_blocks)
            ]

            # predicted measurements for all targets: (m, n_targets) -> (n_targets, m)
            mu = (measurement_matrix @ self.get_point_estimate()).T  # shape (T, m)

            # inverse covariances per target: shape (T, m, m)
            VI = stack(
                [
                    linalg.inv(
                        measurement_matrix @ C @ measurement_matrix.T + cov_mat_meas
                    )
                    for C in C_blocks_list
                ],
                axis=0,
            )

            # differences: (T, N, m) where N = number of measurements
            diff = measurements.T[None, :, :] - mu[:, None, :]  # (T, N, m)

            # squared distances: (T, N)
            d2 = einsum("tnm,tmk,tnk->tn", diff, VI, diff)
            dists = sqrt(d2)

            dists = dists**2  # Squared Mahalanobis distances are usually considered
            # Filter out all that are incompatible with all tracks
            measurements = measurements[:, any(dists < gating_threshold, axis=0)]

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

        # Add jitter to sigma_s for numerical stability
        jitter = 1e-6 * eye(sigma_s.shape[0])
        sigma_s = sigma_s + jitter

        x_posterior = self.x + sigma_xs @ linalg.solve(sigma_s, pseudo_meas - mu_s)
        self.C = self.C - sigma_xs @ linalg.solve(sigma_s, sigma_xs.T)
        self.x = x_posterior

        if self.log_posterior_estimates:
            self.store_posterior_estimates()

    @staticmethod
    def gen_test_points(measurements, kernel_width):
        assert (
            pyrecest.backend.__backend_name__ != "pytorch"
        ), "Not supported on this backend"
        meas_dim = measurements.shape[0]
        n_meas = measurements.shape[1]

        # Use sigma points to generate test points
        all_sample_points_list = [
            JulierSigmaPoints(meas_dim).sigma_points(
                measurements[:, i], sqrt(kernel_width) * eye(meas_dim)
            )
            for i in range(n_meas)
        ]

        return vstack(all_sample_points_list).T

    @staticmethod
    def calc_pseudo_meas(testPoints, measurements, kernel_width):
        assert (
            pyrecest.backend.__backend_name__ != "jax"
        ), "Not supported on this backend"
        n_meas = measurements.shape[1]
        meas_dim = measurements.shape[0]
        n_test_points = testPoints.shape[1]
        pseudo_meas = zeros(n_test_points)
        for i in range(n_test_points):
            a_i = testPoints[:, i]
            for j in range(n_meas):
                gaussian_curr = GaussianDistribution(
                    measurements[:, j], kernel_width * eye(meas_dim)
                )
                pseudo_meas[i] += gaussian_curr.pdf(a_i)
        return pseudo_meas

    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,too-many-statements,too-many-branches
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
        """
        Compute mu_s, Sigma_s, Sigma_xs according to the
        Kernel SME multi-detection + clutter equations in the paper.

        x_prior is stacked [x_1; ...; x_N], C_prior full covariance.
        lambdaMultimeas = lambda^l (Poisson mean number of detections per target).
        falseAlarmRate  = lambda^c (Poisson mean number of clutter points per scan).
        """
        if clutterCov is None:
            clutterCov = zeros_like(covMatMeas)

        meas_dim = covMatMeas.shape[0]
        n_testpoints = testPoints.shape[1]

        # Ensure lambdaMultimeas is a vector of length n_targets
        if isinstance(lambdaMultimeas, (float, int)):
            lambda_vec = float(lambdaMultimeas) * ones(n_targets)
        else:
            lambda_vec = array(lambdaMultimeas).reshape(-1)
            assert (
                lambda_vec.shape[0] == n_targets
            ), "lambdaMultimeas must be scalar or length n_targets"

        lam_c = float(falseAlarmRate)

        state_dim = x_prior.shape[0] // n_targets

        # Per-target state blocks and block covariances
        C_blocks_list = [
            C_prior[
                i * state_dim : (i + 1) * state_dim,  # noqa: E203
                i * state_dim : (i + 1) * state_dim,  # noqa: E203
            ]
            for i in range(n_targets)
        ]
        x_prior_mat = reshape(x_prior, (state_dim, n_targets), order="F")

        I_meas = eye(meas_dim)

        # Per-target predicted measurement means and covariances
        meas_cov_kernel = []  # H C_l H^T + R + Gamma I
        meas_cov_half_kernel = []  # H C_l H^T + R + 0.5 Gamma I
        meas_means = []

        for k in range(n_targets):
            S = (
                measurement_matrix @ C_blocks_list[k] @ measurement_matrix.T
                + covMatMeas
            )
            meas_cov_kernel.append(S + kernel_width * I_meas)
            meas_cov_half_kernel.append(S + 0.5 * kernel_width * I_meas)
            meas_means.append(measurement_matrix @ x_prior_mat[:, k])

        # P_target[i, k] = P_l^Gamma(a_i) for target l=k at testpoint i
        P_target = empty((n_testpoints, n_targets))
        for i in range(n_testpoints):
            z = testPoints[:, i]
            for k in range(n_targets):
                gd_curr = GaussianDistribution(meas_means[k], meas_cov_kernel[k])
                P_target[i, k] = gd_curr.pdf(z)

        # Clutter pdf at testpoints: N(a_i; 0, clutterCov + Gamma I)
        clutter_cov_kernel = clutterCov + kernel_width * I_meas
        clutter_pdf = empty(n_testpoints)
        gd_clutter = GaussianDistribution(zeros(meas_dim), clutter_cov_kernel)
        for i in range(n_testpoints):
            clutter_pdf[i] = gd_clutter.pdf(testPoints[:, i])

        # Mean of pseudo-measurements:
        # mu_s[i] = sum_l lambda_l P_l^Gamma(a_i) + lambda_c * clutter_pdf[i]
        mu_s = P_target @ lambda_vec + lam_c * clutter_pdf

        # Second-order target term:
        # Pij[i, j, k] = P_l^{Gamma/2}((a_i + a_j)/2)
        Pij = zeros((n_testpoints, n_testpoints, n_targets))
        for k in range(n_targets):
            gd_curr = GaussianDistribution(meas_means[k], meas_cov_half_kernel[k])
            for i in range(n_testpoints):
                for j in range(n_testpoints):
                    midpoint = 0.5 * (testPoints[:, i] + testPoints[:, j])
                    Pij[i, j, k] = gd_curr.pdf(midpoint)

        # Covariance of pseudo-measurements Sigma_s
        sigma_s = zeros((n_testpoints, n_testpoints))
        for i in range(n_testpoints):
            for j in range(i, n_testpoints):
                # --- term 1: sum_t lambda_t P_t(i) sum_{m != t} lambda_m P_m(j) ---
                term1 = 0.0
                sum_lambda_P_j = dot(lambda_vec, P_target[j, :])  # noqa: E203
                for t in range(n_targets):
                    term1 += (
                        lambda_vec[t]
                        * P_target[i, t]
                        * (sum_lambda_P_j - lambda_vec[t] * P_target[j, t])
                    )

                # Gaussian kernel between test points: N(a_i; a_j, 2 Gamma I)
                gd_curr = GaussianDistribution(
                    testPoints[:, j], 2.0 * kernel_width * I_meas
                )
                kernel_between = gd_curr.pdf(testPoints[:, i])

                # --- term 2: kernel_between * sum_l (lambda_l^2 P_l^{Gamma/2}(midpoint)) ---
                term2 = kernel_between * sum(
                    (lambda_vec**2) * Pij[i, j, :]
                )  # noqa: E203

                # --- clutter-related terms (3)–(6) ---
                clutter_i = clutter_pdf[i]
                clutter_j = clutter_pdf[j]
                midpoint = 0.5 * (testPoints[:, i] + testPoints[:, j])
                gd_clutter = GaussianDistribution(zeros(meas_dim), clutter_cov_kernel)
                clutter_mid_pdf = gd_clutter.pdf(midpoint)

                term3 = (lam_c**2) * clutter_i * clutter_j
                term4 = mu_s[i] * lam_c * clutter_j
                term5 = mu_s[j] * lam_c * clutter_i
                term6 = lam_c * kernel_between * clutter_mid_pdf

                # Full Sigma^{s_i, s_j}
                sigma_s[i, j] = (
                    term1 + term2 + term3 + term4 + term5 + term6 - mu_s[i] * mu_s[j]
                )

        # Symmetrize Sigma_s
        iu, ju = triu_indices(n_testpoints, k=1)
        sigma_s[ju, iu] = sigma_s[iu, ju]

        # Cross-covariance Sigma_xs
        sigma_xs = zeros((x_prior.shape[0], n_testpoints))

        # Block columns of C_prior (for K^l)
        C_k_full_columns = hsplit(C_prior, n_targets)
        K_list = []
        for k in range(n_targets):
            S_full = (
                measurement_matrix @ C_blocks_list[k] @ measurement_matrix.T
                + covMatMeas
                + kernel_width * I_meas
            )
            K_list.append(
                C_k_full_columns[k] @ measurement_matrix.T @ linalg.inv(S_full)
            )

        # Sigma^{x, s_i}
        for i in range(n_testpoints):
            z = testPoints[:, i]
            # target contributions
            for k in range(n_targets):
                sigma_xs[:, i] += (
                    lambda_vec[k]
                    * P_target[i, k]
                    * (
                        x_prior
                        + K_list[k] @ (z - measurement_matrix @ x_prior_mat[:, k])
                    )
                )
            # clutter term + centering
            sigma_xs[:, i] += x_prior * lam_c * clutter_pdf[i]
            sigma_xs[:, i] -= x_prior * mu_s[i]

        return mu_s, sigma_s, sigma_xs
