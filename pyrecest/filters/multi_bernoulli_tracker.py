# pylint: disable=no-name-in-module,no-member,duplicate-code
import copy
from math import log
from numbers import Integral, Real

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    argmax,
    array,
    empty,
    full,
    linalg,
    ndim,
    stack,
)
from pyrecest.distributions import GaussianDistribution
from scipy.linalg import pinv
from scipy.stats import chi2

from .abstract_multitarget_tracker import AbstractMultitargetTracker
from .kalman_filter import KalmanFilter
from .manifold_mixins import EuclideanFilterMixin
from .track_manager import solve_global_nearest_neighbor


class BernoulliComponent:
    """A Bernoulli component used by :class:`MultiBernoulliTracker`.

    Parameters
    ----------
    existence_probability : float
        Probability that the component exists.
    single_target_state
        Either a single-target filter or a state distribution from which a
        :class:`KalmanFilter` can be instantiated.
    label : hashable, optional
        Persistent track label. If omitted, :class:`MultiBernoulliTracker`
        assigns a monotonically increasing integer label automatically when the
        component becomes active.
    """

    def __init__(self, existence_probability, single_target_state, label=None):
        if not 0.0 <= float(existence_probability) <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")

        self.existence_probability = float(existence_probability)
        self.label = copy.deepcopy(label)

        if isinstance(single_target_state, EuclideanFilterMixin):
            self.single_target_filter = copy.deepcopy(single_target_state)
        else:
            self.single_target_filter = KalmanFilter(single_target_state)

    @property
    def dim(self):
        """Return the state dimension of the component."""
        return self.single_target_filter.dim

    @property
    def filter_state(self):
        """Return the underlying single-target filter state."""
        return self.single_target_filter.filter_state

    def get_point_estimate(self):
        """Return the single-target point estimate."""
        return self.single_target_filter.get_point_estimate()


class MultiBernoulliTracker(AbstractMultitargetTracker):
    """Approximate multi-Bernoulli tracker for linear/Gaussian models.

    The exact multi-Bernoulli update generally yields a multi-Bernoulli mixture.
    To keep the implementation lightweight and close to the rest of PyRecEst, this
    tracker retains a single multi-Bernoulli posterior via a best-assignment
    approximation similar in spirit to the existing nearest-neighbor tracker.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        initial_prior=None,
        tracker_param=None,
        birth_components=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        if tracker_param is None:
            tracker_param = {
                "survival_probability": 0.99,
                "detection_probability": 0.95,
                "clutter_intensity": 1e-9,
                "gating_probability": 0.999,
                "gating_distance_threshold": None,
                "pruning_threshold": 1e-4,
                "maximum_number_of_components": None,
                "birth_existence_probability": 0.8,
                "birth_covariance": None,
                "measurement_to_state_matrix": None,
            }

        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

        self.tracker_param = tracker_param
        self._next_label = 0
        self.birth_components = []
        if birth_components is not None:
            self.birth_components = self._normalize_components(birth_components)

        self.bernoulli_components = []
        if initial_prior is not None:
            self.filter_state = initial_prior

    @staticmethod
    def _clip_probability(probability, eps=1e-12):
        return min(max(float(probability), eps), 1.0 - eps)

    @staticmethod
    def _is_existence_state_tuple(component):
        return (
            isinstance(component, tuple)
            and len(component) == 2
            and isinstance(component[0], Real)
        )

    @staticmethod
    def _is_labeled_existence_state_tuple(component):
        return (
            isinstance(component, tuple)
            and len(component) == 3
            and isinstance(component[0], Real)
        )

    @classmethod
    def _normalize_component(cls, component):
        if isinstance(component, BernoulliComponent):
            return copy.deepcopy(component)
        if cls._is_labeled_existence_state_tuple(component):
            return BernoulliComponent(component[0], component[1], label=component[2])
        if cls._is_existence_state_tuple(component):
            return BernoulliComponent(component[0], component[1])
        return BernoulliComponent(1.0, component)

    @classmethod
    def _normalize_components(cls, components):
        return [cls._normalize_component(component) for component in components]

    def _assign_missing_labels(self, components):
        used_labels = set()
        integer_labels = []

        for component in components:
            if component.label is None:
                continue

            try:
                hash(component.label)
            except TypeError as exc:
                raise TypeError("Track labels must be hashable.") from exc

            if component.label in used_labels:
                raise ValueError("Track labels must be unique among active components.")

            used_labels.add(component.label)
            if isinstance(component.label, Integral):
                integer_labels.append(int(component.label))

        if integer_labels:
            self._next_label = max(self._next_label, max(integer_labels) + 1)

        for component in components:
            if component.label is not None:
                continue

            while self._next_label in used_labels:
                self._next_label += 1

            component.label = self._next_label
            used_labels.add(component.label)
            self._next_label += 1

    @staticmethod
    def _get_component_probability(probability, index):
        if isinstance(probability, (list, tuple)):
            return MultiBernoulliTracker._clip_probability(probability[index])

        try:
            probability_ndim = ndim(probability)
        except (TypeError, AttributeError):
            probability_ndim = None

        if probability_ndim is not None and probability_ndim > 0:
            return MultiBernoulliTracker._clip_probability(probability[index])

        return MultiBernoulliTracker._clip_probability(probability)

    @staticmethod
    def _get_measurement_covariance(cov_mats_meas, measurement_index):
        if cov_mats_meas.ndim == 2:
            return cov_mats_meas
        return cov_mats_meas[:, :, measurement_index]

    @staticmethod
    def _get_state_mean_and_covariance(component):
        state = component.filter_state

        if hasattr(state, "mean"):
            state_mean = state.mean()
        else:
            state_mean = component.get_point_estimate()

        if hasattr(state, "covariance"):
            state_covariance = state.covariance()
        elif hasattr(state, "C"):
            state_covariance = state.C
        else:
            raise ValueError(
                "The single-target state must provide a covariance for association."
            )

        return state_mean, state_covariance

    def _get_gating_distance_threshold(self, measurement_dimension):
        if self.tracker_param["gating_distance_threshold"] is not None:
            return self.tracker_param["gating_distance_threshold"]
        return chi2.ppf(self.tracker_param["gating_probability"], measurement_dimension)

    def _predicted_measurement_moments(
        self, component, measurement_matrix, measurement_covariance
    ):
        state_mean, state_covariance = self._get_state_mean_and_covariance(component)
        predicted_measurement = measurement_matrix @ state_mean
        innovation_covariance = (
            measurement_matrix @ state_covariance @ measurement_matrix.T
            + measurement_covariance
        )
        return predicted_measurement, innovation_covariance

    def _measurement_likelihood_and_distance(
        self, component, measurement, measurement_matrix, measurement_covariance
    ):
        predicted_measurement, innovation_covariance = (
            self._predicted_measurement_moments(
                component, measurement_matrix, measurement_covariance
            )
        )
        innovation = measurement - predicted_measurement
        mahalanobis_distance_squared = float(
            innovation.T @ linalg.solve(innovation_covariance, innovation)
        )
        likelihood = float(
            GaussianDistribution(predicted_measurement, innovation_covariance).pdf(
                measurement
            )
        )
        return likelihood, mahalanobis_distance_squared

    # pylint: disable=too-many-locals
    def _build_assignment_costs(self, measurements, measurement_matrix, cov_mats_meas):
        num_components = self.get_number_of_components()
        num_measurements = measurements.shape[1]
        invalid_cost = 1e12
        detection_cost_matrix = full((num_components, num_measurements), invalid_cost)
        missed_detection_costs = full((num_components,), invalid_cost)
        gating_distance_threshold = self._get_gating_distance_threshold(
            measurements.shape[0]
        )
        clutter_intensity = max(float(self.tracker_param["clutter_intensity"]), 1e-12)

        for i, component in enumerate(self.bernoulli_components):
            predicted_existence = self._clip_probability(
                component.existence_probability
            )
            detection_probability = self._get_component_probability(
                self.tracker_param["detection_probability"], i
            )

            missed_detection_weight = max(
                1.0 - predicted_existence * detection_probability, 1e-12
            )
            missed_detection_costs[i] = -log(missed_detection_weight)

            for j in range(num_measurements):
                measurement_covariance = self._get_measurement_covariance(
                    cov_mats_meas, j
                )
                likelihood, mahalanobis_distance_squared = (
                    self._measurement_likelihood_and_distance(
                        component,
                        measurements[:, j],
                        measurement_matrix,
                        measurement_covariance,
                    )
                )
                if mahalanobis_distance_squared <= gating_distance_threshold:
                    detection_weight = (
                        predicted_existence
                        * detection_probability
                        * likelihood
                        / clutter_intensity
                    )
                    detection_cost_matrix[i, j] = -log(max(detection_weight, 1e-12))

        return detection_cost_matrix, missed_detection_costs

    @staticmethod
    def _get_state_birth_mean(
        measurement, measurement_matrix, measurement_to_state_matrix
    ):
        if callable(measurement_to_state_matrix):
            return measurement_to_state_matrix(measurement, measurement_matrix)
        if measurement_to_state_matrix is None:
            measurement_to_state_matrix = pinv(measurement_matrix)
        return measurement_to_state_matrix @ measurement

    def _create_birth_component_from_measurement(
        self, measurement, measurement_matrix, measurement_covariance
    ):
        birth_covariance = self.tracker_param["birth_covariance"]
        if birth_covariance is None:
            return None

        if callable(birth_covariance):
            birth_covariance = birth_covariance(
                measurement, measurement_matrix, measurement_covariance
            )

        state_mean = self._get_state_birth_mean(
            measurement,
            measurement_matrix,
            self.tracker_param["measurement_to_state_matrix"],
        )

        return BernoulliComponent(
            self.tracker_param["birth_existence_probability"],
            GaussianDistribution(state_mean, birth_covariance),
        )

    @property
    def dim(self):
        if not self.bernoulli_components:
            raise ValueError(
                "Cannot provide the state dimension if no Bernoulli components exist."
            )
        return self.bernoulli_components[0].dim

    @property
    def filter_state(self):
        return copy.deepcopy(self.bernoulli_components)

    @filter_state.setter
    def filter_state(self, new_state):
        self.bernoulli_components = self._normalize_components(new_state)
        self._assign_missing_labels(self.bernoulli_components)
        if self.log_prior_estimates:
            self.store_prior_estimates()

    def get_number_of_components(self):
        """Return the number of Bernoulli components."""
        return len(self.bernoulli_components)

    def get_existence_probabilities(self):
        """Return the existence probabilities of all Bernoulli components."""
        return array(
            [component.existence_probability for component in self.bernoulli_components]
        )

    def get_cardinality_distribution(self):
        """Return the cardinality PMF implied by the Bernoulli components."""
        cardinality_pmf = [1.0]
        for component in self.bernoulli_components:
            existence_probability = float(component.existence_probability)
            new_cardinality_pmf = [0.0] * (len(cardinality_pmf) + 1)
            for cardinality, probability in enumerate(cardinality_pmf):
                new_cardinality_pmf[cardinality] += (
                    1.0 - existence_probability
                ) * probability
                new_cardinality_pmf[cardinality + 1] += (
                    existence_probability * probability
                )
            cardinality_pmf = new_cardinality_pmf
        return array(cardinality_pmf)

    def get_expected_number_of_targets(self):
        """Return the expected cardinality of the multi-Bernoulli posterior."""
        return sum(
            component.existence_probability for component in self.bernoulli_components
        )

    def get_number_of_targets(self):
        """Return the MAP cardinality estimate."""
        if not self.bernoulli_components:
            return 0
        return int(argmax(self.get_cardinality_distribution()))

    def _select_components_for_extraction(self, number_of_targets):
        return sorted(
            self.bernoulli_components,
            key=lambda component: component.existence_probability,
            reverse=True,
        )[:number_of_targets]

    def get_component_labels(self):
        """Return the labels of all active Bernoulli components."""
        return [
            copy.deepcopy(component.label) for component in self.bernoulli_components
        ]

    def get_labeled_components(self, copy_components=True):
        """Return active Bernoulli components keyed by track label."""
        return {
            copy.deepcopy(component.label): (
                copy.deepcopy(component) if copy_components else component
            )
            for component in self.bernoulli_components
        }

    def get_component_by_label(self, label, copy_component=True):
        """Return a Bernoulli component by track label."""
        for component in self.bernoulli_components:
            if component.label == label:
                return copy.deepcopy(component) if copy_component else component
        raise KeyError(f"No Bernoulli component with label {label!r} exists.")

    def get_track_labels(self, number_of_targets=None):
        """Return the labels of the extracted target states."""
        labels, _ = self.get_labeled_point_estimate(
            flatten_vector=False,
            number_of_targets=number_of_targets,
        )
        return labels

    def get_point_estimate(self, flatten_vector=False, number_of_targets=None):
        """Return extracted target states.

        The state extraction follows a common multi-Bernoulli convention: the MAP
        cardinality is used, and the states of the Bernoulli components with the
        highest existence probabilities are returned.
        """
        _, point_estimates = self.get_labeled_point_estimate(
            flatten_vector=flatten_vector,
            number_of_targets=number_of_targets,
        )
        return point_estimates

    def get_labeled_point_estimate(self, flatten_vector=False, number_of_targets=None):
        """Return extracted target states together with persistent labels."""
        if not self.bernoulli_components:
            point_estimates = array([]) if flatten_vector else empty((0, 0))
            return [], point_estimates

        if number_of_targets is None:
            number_of_targets = self.get_number_of_targets()

        if number_of_targets <= 0:
            point_estimates = empty((self.dim, 0))
            labels = []
        else:
            selected_components = self._select_components_for_extraction(
                number_of_targets
            )
            labels = [
                copy.deepcopy(component.label) for component in selected_components
            ]
            point_estimates = stack(
                [component.get_point_estimate() for component in selected_components],
                axis=1,
            )

        if flatten_vector:
            point_estimates = point_estimates.flatten()

        return labels, point_estimates

    def prune(self, pruning_threshold=None):
        """Remove Bernoulli components with low existence probability."""
        if pruning_threshold is None:
            pruning_threshold = self.tracker_param["pruning_threshold"]

        self.bernoulli_components = [
            component
            for component in self.bernoulli_components
            if component.existence_probability >= pruning_threshold
        ]

    def cap(self, maximum_number_of_components=None):
        """Keep only the Bernoulli components with the highest existence probability."""
        if maximum_number_of_components is None:
            maximum_number_of_components = self.tracker_param[
                "maximum_number_of_components"
            ]

        if maximum_number_of_components is None:
            return

        if len(self.bernoulli_components) <= maximum_number_of_components:
            return

        self.bernoulli_components = sorted(
            self.bernoulli_components,
            key=lambda component: component.existence_probability,
            reverse=True,
        )[:maximum_number_of_components]

    def predict_linear(
        self, system_matrices, sys_noises, inputs=None, birth_components=None
    ):
        """Predict all Bernoulli components with a linear/Gaussian model."""
        if isinstance(sys_noises, GaussianDistribution):
            assert backend_all(sys_noises.mu == 0)
            sys_noises = sys_noises.C

        for i, component in enumerate(self.bernoulli_components):
            current_system_matrix = system_matrices
            current_sys_noise = sys_noises
            current_input = inputs

            if system_matrices is not None and ndim(system_matrices) == 3:
                current_system_matrix = system_matrices[:, :, i]
            if sys_noises is not None and ndim(sys_noises) == 3:
                current_sys_noise = sys_noises[:, :, i]
            if inputs is not None and ndim(inputs) == 2:
                current_input = inputs[:, i]

            component.single_target_filter.predict_linear(
                current_system_matrix,
                current_sys_noise,
                current_input,
            )
            component.existence_probability *= self._get_component_probability(
                self.tracker_param["survival_probability"], i
            )
            component.existence_probability = self._clip_probability(
                component.existence_probability
            )

        active_birth_components = self.birth_components
        if birth_components is not None:
            active_birth_components = self._normalize_components(birth_components)

        self.bernoulli_components.extend(copy.deepcopy(active_birth_components))
        self._assign_missing_labels(self.bernoulli_components)
        self.prune()
        self.cap()

        if self.log_prior_estimates:
            self.store_prior_estimates()

    def find_association(self, measurements, measurement_matrix, cov_mats_meas):
        """Find the best measurement-to-Bernoulli association."""
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"

        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, 1)

        if self.get_number_of_components() == 0:
            return array([])

        detection_cost_matrix, missed_detection_costs = self._build_assignment_costs(
            measurements, measurement_matrix, cov_mats_meas
        )
        association_result = solve_global_nearest_neighbor(
            detection_cost_matrix,
            unassigned_track_cost=missed_detection_costs,
            unassigned_measurement_cost=0.0,
            invalid_cost=1e12,
            dummy_dummy_cost=0.0,
        )

        association = full((self.get_number_of_components(),), measurements.shape[1])
        for component_index, measurement_index in association_result.matches:
            association[component_index] = measurement_index
        return association

    # pylint: disable=too-many-locals
    def update_linear(self, measurements, measurement_matrix, cov_mats_meas):
        """Update the multi-Bernoulli tracker with linear/Gaussian measurements."""
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"

        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, 1)

        detection_probability = self.tracker_param["detection_probability"]
        clutter_intensity = max(float(self.tracker_param["clutter_intensity"]), 1e-12)
        num_measurements = measurements.shape[1]
        assigned_measurements = set()

        if self.get_number_of_components() > 0:
            association = self.find_association(
                measurements, measurement_matrix, cov_mats_meas
            )
        else:
            association = array([])

        for i, component in enumerate(self.bernoulli_components):
            current_detection_probability = self._get_component_probability(
                detection_probability, i
            )
            predicted_existence = self._clip_probability(
                component.existence_probability
            )
            assigned_column = int(association[i]) if association.size > i else -1

            if assigned_column < num_measurements:
                measurement_covariance = self._get_measurement_covariance(
                    cov_mats_meas, assigned_column
                )
                likelihood, _ = self._measurement_likelihood_and_distance(
                    component,
                    measurements[:, assigned_column],
                    measurement_matrix,
                    measurement_covariance,
                )
                component.single_target_filter.update_linear(
                    measurements[:, assigned_column],
                    measurement_matrix,
                    measurement_covariance,
                )
                component.existence_probability = (
                    predicted_existence * current_detection_probability * likelihood
                ) / (
                    clutter_intensity
                    + predicted_existence * current_detection_probability * likelihood
                )
                assigned_measurements.add(assigned_column)
            else:
                denominator = 1.0 - predicted_existence * current_detection_probability
                component.existence_probability = (
                    predicted_existence * (1.0 - current_detection_probability)
                ) / max(denominator, 1e-12)

            component.existence_probability = self._clip_probability(
                component.existence_probability
            )

        for j in range(num_measurements):
            if j in assigned_measurements:
                continue
            measurement_covariance = self._get_measurement_covariance(cov_mats_meas, j)
            birth_component = self._create_birth_component_from_measurement(
                measurements[:, j],
                measurement_matrix,
                measurement_covariance,
            )
            if birth_component is not None:
                self.bernoulli_components.append(birth_component)

        self._assign_missing_labels(self.bernoulli_components)
        self.prune()
        self.cap()

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
