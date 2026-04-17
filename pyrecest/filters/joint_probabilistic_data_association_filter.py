# pylint: disable=redefined-builtin,no-name-in-module,no-member,duplicate-code

"""Linear-Gaussian Joint Probabilistic Data Association Filter."""

import warnings
from math import log, pi

import numpy as _np
import pyrecest.backend
from pyrecest.distributions import GaussianDistribution
from scipy.special import logsumexp
from scipy.stats import chi2

from .abstract_nearest_neighbor_tracker import AbstractNearestNeighborTracker


class JointProbabilisticDataAssociationFilter(AbstractNearestNeighborTracker):
    """Joint probabilistic data association for linear-Gaussian multitarget tracking.

    The implementation enumerates all feasible joint association events exactly.
    This is appropriate for a modest number of targets and measurements and keeps the
    code compact and close to the conventions already used by the current tracker
    classes in PyRecEst.
    """

    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        default_association_param = {
            "detection_probability": 0.95,
            "clutter_intensity": 1e-3,
            "gating_distance_threshold": None,
            "max_enumerated_events": 100000,
        }

        if association_param is None:
            association_param = default_association_param
        else:
            association_param = {
                **default_association_param,
                **association_param,
            }

        super().__init__(
            initial_prior,
            association_param,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

        self.latest_association_probabilities = None
        self.latest_map_association = None
        self._latest_posterior_hypotheses = None
        self.filter_bank = []

        if initial_prior is not None:
            self.filter_state = initial_prior

    @staticmethod
    def _get_measurement_covariance(cov_mats_meas, measurement_index):
        if cov_mats_meas.ndim == 2:
            return cov_mats_meas
        if cov_mats_meas.ndim == 3:
            return cov_mats_meas[:, :, measurement_index]
        raise ValueError(
            "cov_mats_meas must have shape (dim_meas, dim_meas) or "
            "(dim_meas, dim_meas, n_meas)"
        )

    @staticmethod
    def _prepare_clutter_intensity(clutter_intensity, n_meas):
        if _np.isscalar(clutter_intensity):
            clutter_intensity = _np.full(n_meas, float(clutter_intensity))
        else:
            clutter_intensity = _np.asarray(clutter_intensity, dtype=float).reshape(-1)
            if clutter_intensity.shape[0] != n_meas:
                raise ValueError(
                    "If clutter_intensity is not scalar, it must have length n_meas."
                )

        if _np.any(clutter_intensity <= 0.0):
            raise ValueError("clutter_intensity must be strictly positive.")

        return clutter_intensity

    @staticmethod
    def _log_gaussian_likelihood(innovation, innovation_covariance):
        sign, logdet = _np.linalg.slogdet(innovation_covariance)
        if sign <= 0.0:
            raise ValueError("Innovation covariance must be positive definite.")

        mahalanobis_distance = float(
            innovation.T @ _np.linalg.solve(innovation_covariance, innovation)
        )
        log_likelihood = -0.5 * (
            innovation.shape[0] * log(2.0 * pi) + logdet + mahalanobis_distance
        )
        return log_likelihood, mahalanobis_distance

    @staticmethod
    def _kalman_update(mu_prior, cov_prior, measurement, measurement_matrix, meas_cov):
        innovation = measurement - measurement_matrix @ mu_prior
        innovation_covariance = (
            measurement_matrix @ cov_prior @ measurement_matrix.T + meas_cov
        )
        kalman_gain = _np.linalg.solve(
            innovation_covariance.T,
            (cov_prior @ measurement_matrix.T).T,
        ).T
        mu_posterior = mu_prior + kalman_gain @ innovation

        identity_matrix = _np.eye(cov_prior.shape[0])
        cov_posterior = (
            identity_matrix - kalman_gain @ measurement_matrix
        ) @ cov_prior @ (
            identity_matrix - kalman_gain @ measurement_matrix
        ).T + kalman_gain @ meas_cov @ kalman_gain.T
        cov_posterior = 0.5 * (cov_posterior + cov_posterior.T)

        return mu_posterior, cov_posterior, innovation_covariance, innovation

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def find_association_probabilities(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
        """Compute marginal association probabilities.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]
                The first entry contains the marginal probabilities of shape
                ``(n_targets, n_meas + 1)``, where column 0 corresponds to a
                missed detection. The second entry is the most likely joint event,
                encoded as measurement indices and ``-1`` for missed detections.
        """
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"

        n_targets = self.get_number_of_targets()
        if n_targets == 0:
            warnings.warn("Currently, there are zero targets.")
            return _np.zeros((0, 1)), _np.zeros((0,), dtype=int)

        measurements = _np.asarray(measurements, dtype=float)
        measurement_matrix = _np.asarray(measurement_matrix, dtype=float)
        cov_mats_meas = _np.asarray(cov_mats_meas, dtype=float)

        if measurements.ndim != 2:
            raise ValueError("measurements must have shape (dim_meas, n_meas)")

        n_meas = measurements.shape[1]
        if measurement_matrix.shape[0] != measurements.shape[0]:
            raise ValueError(
                "Measurement dimension of measurement_matrix and measurements must match."
            )
        if measurement_matrix.shape[1] != self.dim:
            raise ValueError(
                "State dimension of measurement_matrix and tracker state must match."
            )
        if cov_mats_meas.ndim == 3 and cov_mats_meas.shape[2] != n_meas:
            raise ValueError(
                "If cov_mats_meas has three dimensions, the third dimension must be n_meas."
            )

        detection_probability = float(self.association_param["detection_probability"])
        if not 0.0 < detection_probability < 1.0:
            raise ValueError("detection_probability must satisfy 0 < P_D < 1")

        clutter_intensity = self._prepare_clutter_intensity(
            self.association_param["clutter_intensity"], n_meas
        )

        gating_distance_threshold = self.association_param["gating_distance_threshold"]
        if gating_distance_threshold is None:
            gating_distance_threshold = chi2.ppf(0.999, measurements.shape[0])
        gating_distance_threshold = float(gating_distance_threshold)

        if n_meas == 0:
            association_probabilities = _np.ones((n_targets, 1))
            map_association = -_np.ones(n_targets, dtype=int)
            self.latest_association_probabilities = association_probabilities
            self.latest_map_association = map_association
            self._latest_posterior_hypotheses = [{} for _ in range(n_targets)]
            return association_probabilities, map_association

        log_likelihoods = _np.full((n_targets, n_meas), -_np.inf)
        eligible_measurements = [[] for _ in range(n_targets)]
        posterior_hypotheses = [{} for _ in range(n_targets)]

        for target_index in range(n_targets):
            state = self.filter_bank[target_index].filter_state
            if not isinstance(state, GaussianDistribution):
                raise TypeError(
                    "JPDAF currently requires Gaussian filter states for all tracks."
                )

            for measurement_index in range(n_meas):
                meas_cov = self._get_measurement_covariance(
                    cov_mats_meas, measurement_index
                )
                (
                    mu_posterior,
                    cov_posterior,
                    innovation_covariance,
                    innovation,
                ) = self._kalman_update(
                    state.mu,
                    state.C,
                    measurements[:, measurement_index],
                    measurement_matrix,
                    meas_cov,
                )
                (
                    log_likelihood,
                    mahalanobis_distance,
                ) = self._log_gaussian_likelihood(innovation, innovation_covariance)

                if mahalanobis_distance <= gating_distance_threshold:
                    log_likelihoods[target_index, measurement_index] = log_likelihood
                    eligible_measurements[target_index].append(measurement_index)
                    posterior_hypotheses[target_index][measurement_index] = (
                        mu_posterior,
                        cov_posterior,
                    )

        if warn_on_no_meas_for_track and any(
            len(curr_eligible_measurements) == 0
            for curr_eligible_measurements in eligible_measurements
        ):
            warnings.warn(
                "JPDAF: No measurement was within the gating threshold for at least one target."
            )

        track_order = sorted(
            range(n_targets), key=lambda idx: len(eligible_measurements[idx])
        )
        miss_log_weight = log(1.0 - detection_probability)
        detection_log_weight = log(detection_probability)
        max_enumerated_events = int(self.association_param["max_enumerated_events"])

        curr_assignment = -_np.ones(n_targets, dtype=int)
        used_measurements = _np.zeros(n_meas, dtype=bool)
        event_assignments = []
        event_log_weights = []

        def recurse(order_index, curr_log_weight):
            if len(event_log_weights) >= max_enumerated_events:
                raise RuntimeError(
                    "JPDAF exceeded max_enumerated_events during exact joint-event "
                    "enumeration. Reduce the number of targets or measurements, or "
                    "tighten the gate."
                )

            if order_index == n_targets:
                event_assignments.append(curr_assignment.copy())
                event_log_weights.append(curr_log_weight)
                return

            target_index = track_order[order_index]

            curr_assignment[target_index] = -1
            recurse(order_index + 1, curr_log_weight + miss_log_weight)

            for measurement_index in eligible_measurements[target_index]:
                if not used_measurements[measurement_index]:
                    used_measurements[measurement_index] = True
                    curr_assignment[target_index] = measurement_index
                    recurse(
                        order_index + 1,
                        curr_log_weight
                        + detection_log_weight
                        + log_likelihoods[target_index, measurement_index]
                        - log(clutter_intensity[measurement_index]),
                    )
                    used_measurements[measurement_index] = False

            curr_assignment[target_index] = -1

        recurse(0, 0.0)

        event_log_weights = _np.asarray(event_log_weights, dtype=float)
        normalized_event_weights = _np.exp(
            event_log_weights - logsumexp(event_log_weights)
        )

        association_probabilities = _np.zeros((n_targets, n_meas + 1))
        for event_assignment, event_weight in zip(
            event_assignments, normalized_event_weights
        ):
            for target_index, measurement_index in enumerate(event_assignment):
                if measurement_index == -1:
                    association_probabilities[target_index, 0] += event_weight
                else:
                    association_probabilities[
                        target_index, measurement_index + 1
                    ] += event_weight

        map_association = event_assignments[int(_np.argmax(normalized_event_weights))]

        self.latest_association_probabilities = association_probabilities
        self.latest_map_association = map_association
        self._latest_posterior_hypotheses = posterior_hypotheses

        return association_probabilities, map_association

    def find_association(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
        _, map_association = self.find_association_probabilities(
            measurements,
            measurement_matrix,
            cov_mats_meas,
            warn_on_no_meas_for_track=warn_on_no_meas_for_track,
        )
        return map_association

    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    def update_linear(self, measurements, measurement_matrix, covMatsMeas):
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"

        if len(self.filter_bank) == 0:
            warnings.warn("Currently, there are zero targets")
            return

        association_probabilities, _ = self.find_association_probabilities(
            measurements,
            measurement_matrix,
            covMatsMeas,
        )

        posterior_hypotheses = getattr(self, "_latest_posterior_hypotheses", None)
        n_meas = measurements.shape[1]
        for target_index in range(self.get_number_of_targets()):
            prior_state = self.filter_bank[target_index].filter_state
            posterior_mean = (
                association_probabilities[target_index, 0] * prior_state.mu.copy()
            )

            for measurement_index in range(n_meas):
                curr_beta = association_probabilities[
                    target_index, measurement_index + 1
                ]
                if curr_beta == 0.0:
                    continue

                curr_mean, _ = posterior_hypotheses[target_index][measurement_index]
                posterior_mean += curr_beta * curr_mean

            posterior_covariance = association_probabilities[target_index, 0] * (
                prior_state.C
                + _np.outer(
                    prior_state.mu - posterior_mean,
                    prior_state.mu - posterior_mean,
                )
            )

            for measurement_index in range(n_meas):
                curr_beta = association_probabilities[
                    target_index, measurement_index + 1
                ]
                if curr_beta == 0.0:
                    continue

                curr_mean, curr_cov = posterior_hypotheses[target_index][
                    measurement_index
                ]
                mean_diff = curr_mean - posterior_mean
                posterior_covariance += curr_beta * (
                    curr_cov + _np.outer(mean_diff, mean_diff)
                )

            posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
            self.filter_bank[target_index].filter_state = GaussianDistribution(
                posterior_mean,
                posterior_covariance,
                check_validity=False,
            )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()


JPDAF = JointProbabilisticDataAssociationFilter

__all__ = ["JointProbabilisticDataAssociationFilter", "JPDAF"]
