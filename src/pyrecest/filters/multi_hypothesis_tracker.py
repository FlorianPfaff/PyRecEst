# pylint: disable=no-name-in-module,no-member,duplicate-code,redefined-builtin
import warnings
from builtins import all as builtin_all
from copy import deepcopy
from math import log

from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    argmax,
    array,
    asarray,
    exp,
    full,
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
from pyrecest.utils import murty_k_best_assignments
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

    The tracker can be used as a top-K hypothesis generator rather than only as
    a top-1 estimator.  In addition to the usual best-hypothesis accessors, it
    exposes assignment-history marginals, lagged assignment commitments, optional
    whole-hypothesis reranking, score-temperature calibration, and a
    diversity-aware pruning mode. These utilities are useful when the true
    association sequence tends to remain inside the retained beam but is not
    consistently the highest-scoring branch at the current scan.

    Notes
    -----
    * The implementation currently supports linear-Gaussian measurement updates.
    * Global hypothesis generation is based on Murty's ranked assignment
      algorithm, which scales much better than exhaustive branching when there
      are many ambiguous measurements.
    * New-track initiation and track deletion are not included. Unassigned
      measurements are interpreted as clutter.
    * If ``hypothesis_reranker`` is supplied, it should be a callable or an
      object with a ``score(filter_bank, assignment_history, base_log_weight)``
      method returning an additive log score for the complete hypothesis.
    """

    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
        hypothesis_reranker=None,
        hypothesis_diversity_key=None,
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
            "measurement_log_likelihood_weight": 1.0,
            "detection_log_weight": 1.0,
            "missed_detection_log_weight": 1.0,
            "clutter_log_weight": 1.0,
            "score_temperature": 1.0,
            "pruning_strategy": "top_k",
            "diversity_history_length": 3,
            "max_hypotheses_per_signature": 1,
        }

        self.association_param = default_association_param
        if association_param is not None:
            self.association_param.update(association_param)

        self.hypothesis_reranker = hypothesis_reranker
        self.hypothesis_diversity_key = hypothesis_diversity_key
        self._global_hypotheses = []
        self._global_base_log_weights = array([])
        self._global_log_weights = array([])
        self._global_hypothesis_histories = []
        self._global_filter_bank_histories = []

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
    def global_filter_bank_histories(self):
        return self._global_filter_bank_histories

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
            over the supplied hypotheses is assumed. These are interpreted as
            base model scores; optional reranker scores are added separately
            when effective hypothesis weights are computed.
        """
        if not isinstance(hypotheses, list):
            raise ValueError("hypotheses must be provided as a list")

        if len(hypotheses) == 0:
            self._global_hypotheses = []
            self._global_base_log_weights = array([])
            self._global_log_weights = array([])
            self._global_hypothesis_histories = []
            self._global_filter_bank_histories = []
            return

        if self._looks_like_single_hypothesis(hypotheses):
            hypotheses = [hypotheses]

        filter_banks = [
            self._convert_to_filter_bank(hypothesis) for hypothesis in hypotheses
        ]

        n_targets = len(filter_banks[0])
        if not builtin_all(
            len(filter_bank) == n_targets for filter_bank in filter_banks
        ):
            raise ValueError(
                "All global hypotheses must have the same number of tracks"
            )

        self._global_hypotheses = [
            deepcopy(filter_bank) for filter_bank in filter_banks
        ]
        self._global_hypothesis_histories = [[] for _ in self._global_hypotheses]
        self._global_filter_bank_histories = [[] for _ in self._global_hypotheses]

        if log_weights is None:
            self._global_base_log_weights = array(
                [0.0 for _ in range(len(self._global_hypotheses))]
            )
        else:
            if len(log_weights) != len(self._global_hypotheses):
                raise ValueError(
                    "The number of log-weights must match the number of hypotheses"
                )
            self._global_base_log_weights = array(log_weights)

        self._normalize_base_log_weights()
        self._refresh_effective_log_weights()

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
        """Return normalized effective weights used for ranking hypotheses."""
        if self.get_number_of_global_hypotheses() == 0:
            return array([])
        return exp(self._global_log_weights)

    def get_global_hypothesis_log_weights(self):
        """Return normalized effective log weights used for ranking hypotheses."""
        if self.get_number_of_global_hypotheses() == 0:
            return array([])
        return array(self._global_log_weights)

    def get_global_hypothesis_base_log_weights(self):
        """Return relative base-model log weights before optional reranking."""
        if self.get_number_of_global_hypotheses() == 0:
            return array([])
        return array(self._global_base_log_weights)

    def refresh_hypothesis_scores(self):
        """Recompute effective weights after changing a stateful reranker."""
        self._refresh_effective_log_weights()

    def get_top_hypotheses(
        self,
        k=None,
        copy_hypotheses=True,
        include_weights=False,
        include_histories=False,
    ):
        """Return the top hypotheses sorted by effective log weight.

        Parameters
        ----------
        k : int, optional
            Maximum number of hypotheses to return. If omitted, all retained
            hypotheses are returned.
        copy_hypotheses : bool, default=True
            Whether to deep-copy the filter banks before returning them.
        include_weights : bool, default=False
            If true, return dictionaries with ``weight`` and ``log_weight``.
        include_histories : bool, default=False
            If true, include a copy of each hypothesis assignment history.
        """
        n_hypotheses = self.get_number_of_global_hypotheses()
        if n_hypotheses == 0:
            return []
        if k is None:
            k = n_hypotheses
        order = sorted(
            range(n_hypotheses),
            key=lambda index: float(self._global_log_weights[index]),
            reverse=True,
        )[: int(k)]

        top_hypotheses = []
        for index in order:
            hypothesis = self._global_hypotheses[index]
            if copy_hypotheses:
                hypothesis = deepcopy(hypothesis)

            if include_weights or include_histories:
                record = {"index": index, "hypothesis": hypothesis}
                if include_weights:
                    record["weight"] = float(exp(self._global_log_weights[index]))
                    record["log_weight"] = float(self._global_log_weights[index])
                    record["base_log_weight"] = float(
                        self._global_base_log_weights[index]
                    )
                if include_histories:
                    record["history"] = list(self._global_hypothesis_histories[index])
                top_hypotheses.append(record)
            else:
                top_hypotheses.append(hypothesis)

        return top_hypotheses

    def get_best_hypothesis_margin(self):
        """Return the log-weight gap between the best and second hypothesis."""
        n_hypotheses = self.get_number_of_global_hypotheses()
        if n_hypotheses == 0:
            raise ValueError("Currently, there are zero global hypotheses.")
        if n_hypotheses == 1:
            return float("inf")
        sorted_log_weights = sorted(
            (float(log_weight) for log_weight in self._global_log_weights),
            reverse=True,
        )
        return sorted_log_weights[0] - sorted_log_weights[1]

    def get_hypothesis_entropy(self):
        """Return the entropy of the retained effective hypothesis weights."""
        if self.get_number_of_global_hypotheses() == 0:
            return 0.0
        weights = self.get_global_hypothesis_weights()
        return float(-backend_sum(weights * self._global_log_weights))

    def get_assignment_distribution(self, time_index=None, lag=None):
        """Return a probability mass function over global assignments.

        Parameters
        ----------
        time_index : int, optional
            Assignment-history index to inspect. Negative values follow Python
            indexing. If omitted, the latest assignment is used.
        lag : int, optional
            Number of scans to look back from the latest assignment. ``lag=0``
            is the current scan, ``lag=1`` is the previous scan, and so on.
        """
        history_index = self._resolve_history_index(time_index=time_index, lag=lag)
        distribution = {}
        normalizer = 0.0
        for weight, history in zip(
            self.get_global_hypothesis_weights(), self._global_hypothesis_histories
        ):
            assignment = tuple(int(value) for value in history[history_index])
            probability = float(weight)
            distribution[assignment] = distribution.get(assignment, 0.0) + probability
            normalizer += probability

        if normalizer > 0.0:
            distribution = {
                assignment: probability / normalizer
                for assignment, probability in distribution.items()
            }
        return distribution

    def get_assignment_marginals(self, time_index=None, lag=None):
        """Return per-track assignment marginals at a history index.

        The result is a list of dictionaries. Entry ``i`` maps measurement
        indices to probabilities for track ``i``; missed detections use ``-1``.
        """
        distribution = self.get_assignment_distribution(time_index=time_index, lag=lag)
        if not distribution:
            return []

        n_tracks = len(next(iter(distribution)))
        marginals = [dict() for _ in range(n_tracks)]
        for assignment, probability in distribution.items():
            for track_index, measurement_index in enumerate(assignment):
                track_marginal = marginals[track_index]
                track_marginal[measurement_index] = (
                    track_marginal.get(measurement_index, 0.0) + probability
                )
        return marginals

    def get_lagged_assignment_commitment(self, lag=0, mass_threshold=0.9):
        """Return a delayed whole-assignment commitment if enough mass agrees.

        The returned dictionary always contains the most likely assignment and
        its probability. The ``assignment`` field is set to that assignment only
        when its probability is at least ``mass_threshold``; otherwise it is
        ``None``.
        """
        distribution = self.get_assignment_distribution(lag=lag)
        best_assignment, best_probability = max(
            distribution.items(), key=lambda item: item[1]
        )
        committed_assignment = (
            best_assignment if best_probability >= mass_threshold else None
        )
        return {
            "assignment": committed_assignment,
            "best_assignment": best_assignment,
            "probability": best_probability,
            "distribution": distribution,
        }

    def get_track_assignment_commitments(self, lag=0, mass_threshold=0.9):
        """Return delayed per-track commitments from assignment marginals."""
        marginals = self.get_assignment_marginals(lag=lag)
        committed_assignments = []
        committed_probabilities = []
        for track_marginal in marginals:
            best_assignment, best_probability = max(
                track_marginal.items(), key=lambda item: item[1]
            )
            committed_assignments.append(
                best_assignment if best_probability >= mass_threshold else None
            )
            committed_probabilities.append(best_probability)
        return {
            "assignments": tuple(committed_assignments),
            "probabilities": tuple(committed_probabilities),
            "marginals": marginals,
        }

    def get_lagged_point_estimate(
        self, lag=0, hypothesis_index=None, flatten_vector=False, weighted_average=False
    ):
        """Return a delayed filtered point estimate selected using current weights.

        The stored filter-bank snapshot at ``lag`` is used, while the hypothesis
        selection or averaging weights are the current effective hypothesis
        weights. This provides delayed-decision estimates without forcing an
        immediate top-1 commitment.
        """
        history_index = self._resolve_history_index(lag=lag)

        if weighted_average:
            all_point_estimates = stack(
                [
                    self._point_estimate_from_filter_bank(
                        filter_bank_history[history_index]
                    )
                    for filter_bank_history in self._global_filter_bank_histories
                ],
                axis=2,
            )
            weights = self.get_global_hypothesis_weights().reshape(1, 1, -1)
            point_estimate = backend_sum(all_point_estimates * weights, axis=2)
        else:
            if hypothesis_index is None:
                hypothesis_index = self.get_best_hypothesis_index()
            filter_bank = self._global_filter_bank_histories[int(hypothesis_index)][
                history_index
            ]
            point_estimate = self._point_estimate_from_filter_bank(filter_bank)

        if flatten_vector:
            point_estimate = point_estimate.flatten()
        return point_estimate

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
            if not backend_all(asarray(sys_noises.mu) == 0.0):
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
        new_base_log_weights = []
        new_histories = []
        new_filter_bank_histories = []

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
                    measurements_np,
                    measurement_matrix_np,
                    cov_mats_meas_np,
                )
                new_hypotheses.append(updated_filter_bank)
                new_base_log_weights.append(
                    self._global_base_log_weights[parent_index] + assignment_log_score
                )
                new_histories.append(
                    self._global_hypothesis_histories[parent_index] + [assignment]
                )
                new_filter_bank_histories.append(
                    self._global_filter_bank_histories[parent_index]
                    + [deepcopy(updated_filter_bank)]
                )

        self._global_hypotheses = new_hypotheses
        self._global_base_log_weights = array(new_base_log_weights)
        self._global_hypothesis_histories = new_histories
        self._global_filter_bank_histories = new_filter_bank_histories

        self.prune_hypotheses()

        if self.log_posterior_estimates:
            self.store_posterior_estimates()

    def prune_hypotheses(self):
        if self.get_number_of_global_hypotheses() == 0:
            return

        self._normalize_base_log_weights()
        self._refresh_effective_log_weights()
        surviving_indices = self._select_surviving_indices()

        self._global_hypotheses = [
            self._global_hypotheses[i] for i in surviving_indices
        ]
        self._global_base_log_weights = array(
            [self._global_base_log_weights[i] for i in surviving_indices]
        )
        self._global_hypothesis_histories = [
            self._global_hypothesis_histories[i] for i in surviving_indices
        ]
        self._global_filter_bank_histories = [
            self._global_filter_bank_histories[i] for i in surviving_indices
        ]
        self._normalize_base_log_weights()
        self._refresh_effective_log_weights()

    def _select_surviving_indices(self):
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
        pruning_strategy = str(
            self.association_param.get("pruning_strategy", "top_k")
        ).lower()
        if pruning_strategy in ("diverse", "diverse_top_k"):
            surviving_indices = self._select_diverse_surviving_indices(
                surviving_indices, max_global_hypotheses
            )
        else:
            surviving_indices = surviving_indices[:max_global_hypotheses]

        return surviving_indices

    def _select_diverse_surviving_indices(self, candidate_indices, max_hypotheses):
        history_length = int(self.association_param.get("diversity_history_length", 3))
        max_per_signature = self.association_param.get(
            "max_hypotheses_per_signature", 1
        )
        if max_per_signature is None:
            max_per_signature = max_hypotheses
        max_per_signature = int(max_per_signature)
        if max_per_signature <= 0:
            raise ValueError("max_hypotheses_per_signature must be positive")

        selected_indices = []
        signature_counts = {}
        for index in candidate_indices:
            signature = self._hypothesis_signature(index, history_length)
            signature_counts.setdefault(signature, 0)
            if signature_counts[signature] >= max_per_signature:
                continue
            selected_indices.append(index)
            signature_counts[signature] += 1
            if len(selected_indices) >= max_hypotheses:
                break
        return selected_indices

    def _hypothesis_signature(self, index, history_length):
        if self.hypothesis_diversity_key is not None:
            signature = self.hypothesis_diversity_key(
                self._global_hypotheses[index],
                self._global_hypothesis_histories[index],
            )
        else:
            history = self._global_hypothesis_histories[index]
            if history_length <= 0:
                signature = tuple(history)
            else:
                signature = tuple(history[-history_length:])

        try:
            hash(signature)
        except TypeError:
            signature = repr(signature)
        return signature

    @staticmethod
    def _point_estimate_from_filter_bank(filter_bank):
        return stack(
            [filter_obj.get_point_estimate() for filter_obj in filter_bank],
            axis=1,
        )

    def _normalize_base_log_weights(self):
        if len(self._global_base_log_weights) == 0:
            return
        self._global_base_log_weights = self._global_base_log_weights - backend_max(
            self._global_base_log_weights
        )

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

    def _refresh_effective_log_weights(self):
        if len(self._global_base_log_weights) == 0:
            self._global_log_weights = array([])
            return

        score_temperature = float(self.association_param.get("score_temperature", 1.0))
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")

        effective_log_weights = []
        for index, base_log_weight in enumerate(self._global_base_log_weights):
            reranker_score = self._score_hypothesis_with_reranker(
                self._global_hypotheses[index],
                self._global_hypothesis_histories[index],
                float(base_log_weight),
            )
            effective_log_weights.append(float(base_log_weight) + reranker_score)

        self._global_log_weights = array(effective_log_weights) / score_temperature
        self._normalize_log_weights()

    def _score_hypothesis_with_reranker(
        self, filter_bank, assignment_history, base_log_weight
    ):
        if self.hypothesis_reranker is None:
            return 0.0
        if hasattr(self.hypothesis_reranker, "score"):
            score = self.hypothesis_reranker.score(
                filter_bank,
                assignment_history,
                base_log_weight,
            )
        elif callable(self.hypothesis_reranker):
            score = self.hypothesis_reranker(
                filter_bank,
                assignment_history,
                base_log_weight,
            )
        else:
            raise ValueError(
                "hypothesis_reranker must be callable or expose a score method"
            )
        return float(score)

    def _resolve_history_index(self, time_index=None, lag=None):
        if self.get_number_of_global_hypotheses() == 0:
            raise ValueError("Currently, there are zero global hypotheses.")
        if (
            not self._global_hypothesis_histories
            or len(self._global_hypothesis_histories[0]) == 0
        ):
            raise ValueError("No assignment history is available yet.")

        history_length = len(self._global_hypothesis_histories[0])
        if not builtin_all(
            len(history) == history_length
            for history in self._global_hypothesis_histories
        ):
            raise ValueError(
                "All hypotheses must have assignment histories of equal length"
            )

        if lag is not None:
            if time_index is not None:
                raise ValueError("Specify either time_index or lag, not both")
            if lag < 0:
                raise ValueError("lag must be non-negative")
            resolved_index = history_length - 1 - int(lag)
        else:
            resolved_index = -1 if time_index is None else int(time_index)
            if resolved_index < 0:
                resolved_index += history_length

        if resolved_index < 0 or resolved_index >= history_length:
            raise IndexError("Requested assignment history index is out of range")
        return resolved_index

    def _get_base_log_score(self, n_meas):
        eps = 1.0e-12
        detection_probability = min(
            max(float(self.association_param["detection_probability"]), eps),
            1.0 - eps,
        )
        clutter_intensity = max(float(self.association_param["clutter_intensity"]), eps)
        missed_detection_probability = 1.0 - detection_probability
        missed_detection_weight = float(
            self.association_param.get("missed_detection_log_weight", 1.0)
        )
        clutter_weight = float(self.association_param.get("clutter_log_weight", 1.0))

        n_tracks = self.get_number_of_targets()
        return n_meas * clutter_weight * log(clutter_intensity) + n_tracks * (
            missed_detection_weight * log(missed_detection_probability)
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
        measurement_weight = float(
            self.association_param.get("measurement_log_likelihood_weight", 1.0)
        )
        detection_weight = float(
            self.association_param.get("detection_log_weight", 1.0)
        )
        missed_detection_weight = float(
            self.association_param.get("missed_detection_log_weight", 1.0)
        )
        clutter_weight = float(self.association_param.get("clutter_log_weight", 1.0))

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
                        detection_weight * log(detection_probability)
                        + measurement_weight * log_likelihood
                        - missed_detection_weight * log(missed_detection_probability)
                        - clutter_weight * log(clutter_intensity)
                    )
                    track_candidates.append((j, gain))

            track_candidates.sort(key=lambda item: item[1], reverse=True)
            if max_measurements_per_track is not None:
                track_candidates = track_candidates[:max_measurements_per_track]
            candidate_measurements.append(track_candidates)

        return candidate_measurements

    def _enumerate_candidate_assignments(  # pylint: disable=too-many-locals
        self, candidate_measurements, base_log_score
    ):
        max_assignments = int(
            self.association_param["max_hypotheses_per_global_hypothesis"]
        )
        n_tracks = len(candidate_measurements)
        if n_tracks == 0:
            return [(base_log_score, tuple())]

        unique_measurements = sorted(
            {
                measurement_index
                for track_candidates in candidate_measurements
                for measurement_index, _ in track_candidates
            }
        )
        if len(unique_measurements) == 0:
            return [(base_log_score, tuple(-1 for _ in range(n_tracks)))]

        measurement_to_column = {
            measurement_index: column_index
            for column_index, measurement_index in enumerate(unique_measurements)
        }
        cost_matrix = full(
            (n_tracks, len(unique_measurements)),
            float("inf"),
        )
        for track_index, track_candidates in enumerate(candidate_measurements):
            for measurement_index, gain in track_candidates:
                cost_matrix[track_index, measurement_to_column[measurement_index]] = (
                    -float(gain)
                )

        ranked_assignments = murty_k_best_assignments(
            cost_matrix,
            k=max_assignments,
        )

        best_assignments = []
        for ranked_assignment in ranked_assignments:
            assignment = tuple(
                -1 if column_index < 0 else unique_measurements[int(column_index)]
                for column_index in ranked_assignment["assignment"]
            )
            best_assignments.append(
                (base_log_score - float(ranked_assignment["cost"]), assignment)
            )

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

        if builtin_all(isinstance(item, EuclideanFilterMixin) for item in hypothesis):
            filter_bank = deepcopy(hypothesis)
        else:
            filter_bank = [KalmanFilter(filter_state) for filter_state in hypothesis]

        if not builtin_all(
            id(filter_bank[i]) != id(filter_bank[j])
            for i in range(len(filter_bank))
            for j in range(i + 1, len(filter_bank))
        ):
            raise ValueError(
                "No two filters of a filter bank should have the same handle"
            )

        return filter_bank
