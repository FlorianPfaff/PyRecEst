# pylint: disable=no-name-in-module,no-member,duplicate-code,redefined-builtin
import warnings
from copy import deepcopy
from math import log

from pyrecest.backend import (
    all,
    argmax,
    array,
    asarray,
    exp,
    linalg,
)
from pyrecest.backend import log as backend_log
from pyrecest.backend import max as backend_max
from pyrecest.backend import (
    ndim,
    pi,
    sqrt,
    stack,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.distributions import GaussianDistribution
from scipy.stats import chi2

from .abstract_multitarget_tracker import AbstractMultitargetTracker
from .kalman_filter import KalmanFilter
from .manifold_mixins import EuclideanFilterMixin


class MultiHypothesisTracker(AbstractMultitargetTracker):
    """A simple track-oriented multi-hypothesis tracker.

    This implementation is intentionally lightweight and stays close to the
    conventions of the existing tracker classes in :mod:`pyrecest.filters`.
    It keeps a fixed number of tracks and maintains several competing global
    hypotheses, each represented by a complete bank of single-target filters.

    Notes
    -----
    * The implementation currently supports linear-Gaussian measurement updates.
    * Hypothesis generation uses gated exhaustive branching with pruning. This
      keeps the implementation dependency-free, but it is not meant to replace
      a full Murty-based production MHT.
    * New-track initiation and track deletion are not included. Unassigned
      measurements are interpreted as clutter.
    """

    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

        default_association_param = {
            "distance_metric_pos": "Mahalanobis",
            "square_dist": True,
            "gating_probability": 0.999,
            "gating_distance_threshold": None,
            "detection_probability": 0.99,
            "clutter_intensity": 1.0e-6,
            "max_global_hypotheses": 10,
            "max_hypotheses_per_global_hypothesis": 5,
            "max_measurements_per_track": 5,
            "prune_log_weight_delta": 20.0,
        }

        self.association_param = default_association_param
        if association_param is not None:
            self.association_param.update(association_param)

        self._global_hypotheses = []
        self._global_log_weights = array([])
        self._global_hypothesis_histories = []

        if initial_prior is not None:
            self.filter_state = initial_prior

    @property
    def dim(self) -> int:
        if self.get_number_of_targets() == 0:
            raise ValueError("Cannot provide state dimension if there are no targets.")
        return self.filter_bank[0].dim

    @property
    def global_hypotheses(self):
        return self._global_hypotheses

    @property
    def global_hypothesis_histories(self):
        return self._global_hypothesis_histories

    @property
    def filter_bank(self):
        if self.get_number_of_global_hypotheses() == 0:
            warnings.warn("Currently, there are zero global hypotheses.")
            return []
        return self.get_best_hypothesis()

    @filter_bank.setter
    def filter_bank(self, new_filter_bank):
        self.filter_state = new_filter_bank

    @property
    def filter_state(self):
        if self.get_number_of_targets() == 0:
            warnings.warn("Currently, there are zero targets.")
            return None
        return [filter_obj.filter_state for filter_obj in self.filter_bank]

    @filter_state.setter
    def filter_state(self, new_state):
        if self._looks_like_single_hypothesis(new_state):
            self.set_global_hypotheses([new_state])
        else:
            self.set_global_hypotheses(new_state)

    def set_global_hypotheses(self, hypotheses, log_weights=None):
        """Set the global hypotheses explicitly.

        Parameters
        ----------
        hypotheses : list
            Either a list with one filter bank, or a list of filter banks.
            A filter bank can either contain Euclidean filters directly or
            Gaussian states, which are wrapped in Kalman filters.
        log_weights : array-like, optional
            Log-weights of the global hypotheses. If omitted, a uniform prior
            over the supplied hypotheses is assumed.
        """
        if not isinstance(hypotheses, list):
            raise ValueError("hypotheses must be provided as a list")

        if len(hypotheses) == 0:
            self._global_hypotheses = []
            self._global_log_weights = array([])
            self._global_hypothesis_histories = []
            return

        if self._looks_like_single_hypothesis(hypotheses):
            hypotheses = [hypotheses]

        filter_banks = [
            self._convert_to_filter_bank(hypothesis) for hypothesis in hypotheses
        ]

        n_targets = len(filter_banks[0])
        if not all(len(filter_bank) == n_targets for filter_bank in filter_banks):
            raise ValueError(
                "All global hypotheses must have the same number of tracks"
            )

        self._global_hypotheses = [
            deepcopy(filter_bank) for filter_bank in filter_banks
        ]
        self._global_hypothesis_histories = [[] for _ in self._global_hypotheses]

        if log_weights is None:
            self._global_log_weights = array(
                [0.0 for _ in range(len(self._global_hypotheses))]
            )
        else:
            if len(log_weights) != len(self._global_hypotheses):
                raise ValueError(
                    "The number of log-weights must match the number of hypotheses"
                )
            self._global_log_weights = array(log_weights)

        self._normalize_log_weights()

        if self.log_prior_estimates and self.get_number_of_targets() > 0:
            self.store_prior_estimates()

    def get_number_of_targets(self) -> int:
        if self.get_number_of_global_hypotheses() == 0:
            return 0
        return len(self._global_hypotheses[0])

    def get_number_of_global_hypotheses(self) -> int:
        return len(self._global_hypotheses)

    def get_best_hypothesis_index(self) -> int:
        if self.get_number_of_global_hypotheses() == 0:
            raise ValueError("Currently, there are zero global hypotheses.")
        return int(argmax(self._global_log_weights))

    def get_best_hypothesis(self):
        return self._global_hypotheses[self.get_best_hypothesis_index()]

    def get_global_hypothesis_weights(self):
        if self.get_number_of_global_hypotheses() == 0:
            return array([])
        return exp(self._global_log_weights)

    def get_point_estimate(self, flatten_vector=False, weighted_average=False):
        num_targets = self.get_number_of_targets()
        if num_targets == 0:
            warnings.warn("Currently, there are zero targets.")
            return None

        if weighted_average and self.get_number_of_global_hypotheses() > 1:
            all_point_estimates = stack(
                [
                    stack(
                        [filter_obj.get_point_estimate() for filter_obj in filter_bank],
                        axis=1,
                    )
                    for filter_bank in self._global_hypotheses
                ],
                axis=2,
            )
            weights = self.get_global_hypothesis_weights().reshape(1, 1, -1)
            point_estimate = backend_sum(all_point_estimates * weights, axis=2)
        else:
            point_estimate = stack(
                [filter_obj.get_point_estimate() for filter_obj in self.filter_bank],
                axis=1,
            )

        if flatten_vector:
            point_estimate = point_estimate.flatten()

        return point_estimate

    def predict_linear(self, system_matrices, sys_noises, inputs=None):
        if self.get_number_of_targets() == 0:
            warnings.warn("Currently, there are zero targets.")
            return

        if isinstance(sys_noises, GaussianDistribution):
            if not all(asarray(sys_noises.mu) == 0.0):
                raise ValueError("System noise mean is expected to be zero")
            sys_noises = sys_noises.C

        for filter_bank in self._global_hypotheses:
            curr_sys_matrix = system_matrices
            curr_sys_noise = sys_noises
            curr_input = inputs
            for i, filter_obj in enumerate(filter_bank):
                if system_matrices is not None and ndim(system_matrices) == 3:
                    curr_sys_matrix = system_matrices[:, :, i]
                if sys_noises is not None and ndim(sys_noises) == 3:
                    curr_sys_noise = sys_noises[:, :, i]
                if inputs is not None and ndim(inputs) == 2:
                    curr_input = inputs[:, i]
                filter_obj.predict_linear(curr_sys_matrix, curr_sys_noise, curr_input)

        if self.log_prior_estimates:
            self.store_prior_estimates()

    def update_linear(
        self, measurements, measurement_matrix, cov_mats_meas
    ):  # pylint: disable=too-many-locals
        if self.get_number_of_targets() == 0:
            warnings.warn("Currently, there are zero targets.")
            return

        if self.association_param["distance_metric_pos"].lower() != "mahalanobis":
            raise ValueError("Only Mahalanobis gating is currently supported")

        if self.get_number_of_global_hypotheses() == 0:
            warnings.warn("Currently, there are zero global hypotheses.")
            return

        measurements_np = asarray(measurements)
        measurement_matrix_np = asarray(measurement_matrix)
        cov_mats_meas_np = asarray(cov_mats_meas)

        if measurements_np.ndim != 2:
            raise ValueError("measurements must be a 2D array")
        if measurement_matrix_np.ndim != 2:
            raise ValueError("measurement_matrix must be a 2D array")
        if cov_mats_meas_np.ndim not in (2, 3):
            raise ValueError("cov_mats_meas must be a matrix or a stack of matrices")
        if (
            cov_mats_meas_np.ndim == 3
            and cov_mats_meas_np.shape[2] != measurements_np.shape[1]
        ):
            raise ValueError(
                "If measurement covariances are provided per measurement, the third "
                "dimension must match the number of measurements"
            )

        expected_state_dim = self.filter_bank[0].get_point_estimate().shape[0]
        if measurement_matrix_np.shape[0] != measurements_np.shape[0]:
            raise ValueError(
                "The measurement dimension must match the number of rows in the "
                "measurement matrix"
            )
        if measurement_matrix_np.shape[1] != expected_state_dim:
            raise ValueError(
                "The state dimension must match the number of columns in the "
                "measurement matrix"
            )

        n_meas = measurements_np.shape[1]
        base_log_score = self._get_base_log_score(n_meas)

        new_hypotheses = []
        new_log_weights = []
        new_histories = []

        for parent_index, parent_filter_bank in enumerate(self._global_hypotheses):
            candidate_measurements = self._build_candidate_measurements(
                parent_filter_bank,
                measurements_np,
                measurement_matrix_np,
                cov_mats_meas_np,
            )
            candidate_assignments = self._enumerate_candidate_assignments(
                candidate_measurements,
                base_log_score,
            )

            for assignment_log_score, assignment in candidate_assignments:
                updated_filter_bank = self._apply_assignment(
                    parent_filter_bank,
                    assignment,
                    measurements,
                    measurement_matrix,
                    cov_mats_meas,
                )
                new_hypotheses.append(updated_filter_bank)
                new_log_weights.append(
                    self._global_log_weights[parent_index] + assignment_log_score
                )
                new_histories.append(
                    self._global_hypothesis_histories[parent_index] + [assignment]
                )

        self._global_hypotheses = new_hypotheses
        self._global_log_weights = array(new_log_weights)
        self._global_hypothesis_histories = new_histories

        self.prune_hypotheses()

        if self.log_posterior_estimates:
            self.store_posterior_estimates()

    def prune_hypotheses(self):
        if self.get_number_of_global_hypotheses() == 0:
            return

        max_global_hypotheses = int(self.association_param["max_global_hypotheses"])
        prune_log_weight_delta = float(self.association_param["prune_log_weight_delta"])

        best_log_weight = float(backend_max(self._global_log_weights))
        surviving_indices = [
            i
            for i in range(self.get_number_of_global_hypotheses())
            if (
                float(self._global_log_weights[i])
                >= best_log_weight - prune_log_weight_delta
            )
        ]

        surviving_indices.sort(
            key=lambda i: float(self._global_log_weights[i]), reverse=True
        )
        surviving_indices = surviving_indices[:max_global_hypotheses]

        self._global_hypotheses = [
            self._global_hypotheses[i] for i in surviving_indices
        ]
        self._global_log_weights = array(
            [self._global_log_weights[i] for i in surviving_indices]
        )
        self._global_hypothesis_histories = [
            self._global_hypothesis_histories[i] for i in surviving_indices
        ]
        self._normalize_log_weights()

    def _normalize_log_weights(self):
        if len(self._global_log_weights) == 0:
            return
        max_log_weight = backend_max(self._global_log_weights)
        shifted_weights = exp(self._global_log_weights - max_log_weight)
        self._global_log_weights = (
            self._global_log_weights
            - max_log_weight
            - backend_log(backend_sum(shifted_weights))
        )

    def _get_base_log_score(self, n_meas):
        eps = 1.0e-12
        detection_probability = min(
            max(float(self.association_param["detection_probability"]), eps),
            1.0 - eps,
        )
        clutter_intensity = max(float(self.association_param["clutter_intensity"]), eps)
        missed_detection_probability = 1.0 - detection_probability

        n_tracks = self.get_number_of_targets()
        return n_meas * log(clutter_intensity) + n_tracks * log(
            missed_detection_probability
        )

    def _build_candidate_measurements(  # pylint: disable=too-many-locals
        self,
        filter_bank,
        measurements,
        measurement_matrix,
        cov_mats_meas,
    ):
        n_tracks = len(filter_bank)
        n_meas = measurements.shape[1]
        candidate_measurements = []

        eps = 1.0e-12
        detection_probability = min(
            max(float(self.association_param["detection_probability"]), eps),
            1.0 - eps,
        )
        clutter_intensity = max(float(self.association_param["clutter_intensity"]), eps)
        missed_detection_probability = 1.0 - detection_probability

        gating_distance_threshold = self.association_param["gating_distance_threshold"]
        if gating_distance_threshold is None:
            gating_distance_threshold = chi2.ppf(
                float(self.association_param["gating_probability"]),
                measurements.shape[0],
            )
            if not self.association_param.get("square_dist", True):
                gating_distance_threshold = sqrt(gating_distance_threshold)

        max_measurements_per_track = self.association_param[
            "max_measurements_per_track"
        ]

        for i in range(n_tracks):
            track_candidates = []
            gaussian = filter_bank[i].filter_state
            predicted_measurement = measurement_matrix @ asarray(gaussian.mu)
            for j in range(n_meas):
                meas_cov = self._get_measurement_covariance(cov_mats_meas, j)
                innovation_cov = measurement_matrix @ asarray(
                    gaussian.C
                ) @ measurement_matrix.T + asarray(meas_cov)
                innovation = measurements[:, j] - predicted_measurement
                mahalanobis_squared, log_likelihood = (
                    self._mahalanobis_squared_and_log_likelihood(
                        innovation,
                        innovation_cov,
                    )
                )
                gating_distance = mahalanobis_squared
                if not self.association_param.get("square_dist", True):
                    gating_distance = sqrt(gating_distance)
                if gating_distance <= gating_distance_threshold:
                    gain = (
                        log(detection_probability)
                        + log_likelihood
                        - log(missed_detection_probability)
                        - log(clutter_intensity)
                    )
                    track_candidates.append((j, gain))

            track_candidates.sort(key=lambda item: item[1], reverse=True)
            if max_measurements_per_track is not None:
                track_candidates = track_candidates[:max_measurements_per_track]
            candidate_measurements.append(track_candidates)

        return candidate_measurements

    def _enumerate_candidate_assignments(self, candidate_measurements, base_log_score):
        max_assignments = int(
            self.association_param["max_hypotheses_per_global_hypothesis"]
        )
        n_tracks = len(candidate_measurements)

        best_assignments = []

        optimistic_future_gain = [0.0 for _ in range(n_tracks + 1)]
        for i in range(n_tracks - 1, -1, -1):
            best_local_gain = 0.0
            if len(candidate_measurements[i]) > 0:
                best_local_gain = max(0.0, candidate_measurements[i][0][1])
            optimistic_future_gain[i] = optimistic_future_gain[i + 1] + best_local_gain

        def maybe_store_assignment(assignment_gain, assignment):
            assignment_log_score = base_log_score + assignment_gain
            if len(best_assignments) < max_assignments:
                best_assignments.append((assignment_log_score, assignment))
                best_assignments.sort(key=lambda item: item[0], reverse=True)
                return

            if assignment_log_score > best_assignments[-1][0]:
                best_assignments[-1] = (assignment_log_score, assignment)
                best_assignments.sort(key=lambda item: item[0], reverse=True)

        def recurse(track_index, used_measurements, current_assignment, current_gain):
            if track_index == n_tracks:
                maybe_store_assignment(current_gain, tuple(current_assignment))
                return

            if len(best_assignments) == max_assignments:
                upper_bound = (
                    base_log_score + current_gain + optimistic_future_gain[track_index]
                )
                if upper_bound <= best_assignments[-1][0]:
                    return

            current_assignment.append(-1)
            recurse(
                track_index + 1,
                used_measurements,
                current_assignment,
                current_gain,
            )
            current_assignment.pop()

            for measurement_index, gain in candidate_measurements[track_index]:
                if measurement_index in used_measurements:
                    continue
                used_measurements.add(measurement_index)
                current_assignment.append(measurement_index)
                recurse(
                    track_index + 1,
                    used_measurements,
                    current_assignment,
                    current_gain + gain,
                )
                current_assignment.pop()
                used_measurements.remove(measurement_index)

        recurse(0, set(), [], 0.0)
        return best_assignments

    def _apply_assignment(  # pylint: disable=too-many-positional-arguments
        self,
        filter_bank,
        assignment,
        measurements,
        measurement_matrix,
        cov_mats_meas,
    ):
        updated_filter_bank = deepcopy(filter_bank)
        for track_index, measurement_index in enumerate(assignment):
            if measurement_index < 0:
                continue
            curr_meas_cov = self._get_measurement_covariance(
                cov_mats_meas, measurement_index
            )
            updated_filter_bank[track_index].update_linear(
                measurements[:, measurement_index],
                measurement_matrix,
                curr_meas_cov,
            )
        return updated_filter_bank

    @staticmethod
    def _mahalanobis_squared_and_log_likelihood(innovation, innovation_cov):
        innovation = asarray(innovation)
        innovation_cov = asarray(innovation_cov)

        mahalanobis_squared = float(
            innovation.T @ linalg.solve(innovation_cov, innovation)
        )
        det_val = float(linalg.det(innovation_cov))
        if det_val <= 0.0:
            raise ValueError("Innovation covariance must be positive definite")
        log_det = backend_log(det_val)

        log_likelihood = -0.5 * (
            mahalanobis_squared + innovation.shape[0] * backend_log(2.0 * pi) + log_det
        )
        return mahalanobis_squared, float(log_likelihood)

    @staticmethod
    def _get_measurement_covariance(cov_mats_meas, measurement_index):
        if ndim(asarray(cov_mats_meas)) == 2:
            return cov_mats_meas
        return cov_mats_meas[:, :, measurement_index]

    @staticmethod
    def _looks_like_single_hypothesis(hypotheses):
        if len(hypotheses) == 0:
            return True
        return not isinstance(hypotheses[0], list)

    @staticmethod
    def _convert_to_filter_bank(hypothesis):
        if not isinstance(hypothesis, list):
            raise ValueError("Each hypothesis must be provided as a list")

        if len(hypothesis) == 0:
            return []

        if all(isinstance(item, EuclideanFilterMixin) for item in hypothesis):
            filter_bank = deepcopy(hypothesis)
        else:
            filter_bank = [KalmanFilter(filter_state) for filter_state in hypothesis]

        if not all(
            id(filter_bank[i]) != id(filter_bank[j])
            for i in range(len(filter_bank))
            for j in range(i + 1, len(filter_bank))
        ):
            raise ValueError(
                "No two filters of a filter bank should have the same handle"
            )

        return filter_bank
