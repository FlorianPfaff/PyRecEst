import copy
import warnings

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    all,
    argsort,
    array,
    column_stack,
    dot,
    exp,
    eye,
    hstack,
    linalg,
    ones,
    outer,
    pi,
    stack,
    sum,
    zeros,
    zeros_like,
)
from pyrecest.distributions import GaussianDistribution

from .abstract_multitarget_tracker import AbstractMultitargetTracker


class GaussianMixturePHDState:
    """
    Container for a Gaussian-mixture PHD intensity.

    The weights are intentionally not normalized. Their sum equals the expected
    number of targets.
    """

    def __init__(self, dists: list[GaussianDistribution] | None = None, w=None):
        dists = [] if dists is None else list(dists)
        self.dists = copy.deepcopy(dists)

        if w is None:
            w = zeros((len(self.dists),))

        self.w = array(w)
        if self.w.ndim == 0:
            self.w = self.w.reshape((1,))

        if len(self.dists) != self.w.shape[0]:
            raise ValueError("Number of Gaussian components and weights must match.")

        if any(not isinstance(dist, GaussianDistribution) for dist in self.dists):
            raise TypeError("All components must be GaussianDistribution instances.")

        if len(self.dists) > 1 and any(
            dist.dim != self.dists[0].dim for dist in self.dists[1:]
        ):
            raise ValueError("All Gaussian components must have the same dimension.")

    @property
    def dim(self) -> int:
        return 0 if not self.dists else self.dists[0].dim


class GaussianMixturePHDFilter(
    AbstractMultitargetTracker
):  # pylint: disable=too-many-instance-attributes
    """
    Lightweight Gaussian-mixture PHD filter for linear/Gaussian multitarget tracking.

    This implementation is deliberately small and currently restricted to the numpy
    backend. It supports linear prediction and update models, persistent birth
    components, pruning, Mahalanobis merging, and a simple point-estimate extractor.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        initial_components: list[GaussianDistribution] | None = None,
        initial_weights=None,
        birth_components: list[GaussianDistribution] | None = None,
        birth_weights=None,
        survival_probability: float = 0.99,
        detection_probability: float = 0.9,
        clutter_intensity: float = 1e-6,
        pruning_threshold: float = 1e-5,
        merging_threshold: float = 4.0,
        extraction_threshold: float = 0.5,
        max_components: int = 100,
        log_prior_estimates: bool = True,
        log_posterior_estimates: bool = True,
    ):
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Currently only supported for the numpy backend."

        AbstractMultitargetTracker.__init__(
            self,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

        self.survival_probability = survival_probability
        self.detection_probability = detection_probability
        self.clutter_intensity = clutter_intensity
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold
        self.extraction_threshold = extraction_threshold
        self.max_components = max_components

        self._components: list[GaussianDistribution] = []
        self._weights = zeros((0,))
        self._dim: int | None = None

        self.birth_components: list[GaussianDistribution] = []
        self.birth_weights = zeros((0,))

        if birth_components is not None:
            self.set_birth_model(birth_components, birth_weights)

        if initial_components is not None:
            if initial_weights is None:
                initial_weights = ones((len(initial_components),))
            self.filter_state = GaussianMixturePHDState(
                initial_components,
                initial_weights,
            )

    @property
    def dim(self) -> int:
        if self._dim is None:
            warnings.warn("Filter state is not initialized yet, output 0 as dimension.")
            return 0
        return self._dim

    @property
    def filter_state(self) -> GaussianMixturePHDState:
        return GaussianMixturePHDState(self._components, self._weights)

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, GaussianMixturePHDState):
            components = new_state.dists
            weights = new_state.w
        elif isinstance(new_state, tuple) and len(new_state) == 2:
            components, weights = new_state
        elif isinstance(new_state, list):
            components = new_state
            weights = ones((len(components),))
        else:
            raise ValueError(
                "new_state must be a GaussianMixturePHDState, a "
                "(components, weights) tuple, or a list of Gaussian components."
            )

        state = GaussianMixturePHDState(components, weights)
        self._components = copy.deepcopy(state.dists)
        self._weights = array(state.w)
        if self._weights.ndim == 0:
            self._weights = self._weights.reshape((1,))

        if self._components:
            self._dim = self._components[0].dim

        if self.log_prior_estimates:
            self.store_prior_estimates()

    def set_birth_model(
        self,
        birth_components: list[GaussianDistribution],
        birth_weights=None,
    ):
        if birth_weights is None:
            birth_weights = 0.05 * ones((len(birth_components),))

        birth_state = GaussianMixturePHDState(birth_components, birth_weights)
        self.birth_components = copy.deepcopy(birth_state.dists)
        self.birth_weights = array(birth_state.w)

        if self._dim is None and self.birth_components:
            self._dim = self.birth_components[0].dim

    def get_expected_number_of_targets(self) -> float:
        return 0.0 if self._weights.size == 0 else float(sum(self._weights))

    def _get_extraction_indices(self):
        if len(self._components) == 0:
            return []

        estimated_cardinality = int(round(self.get_expected_number_of_targets()))
        if estimated_cardinality <= 0:
            return []

        sorted_indices = list(argsort(self._weights)[::-1])
        high_weight_indices = [
            int(index)
            for index in sorted_indices
            if self._weights[index] >= self.extraction_threshold
        ]

        if len(high_weight_indices) >= estimated_cardinality:
            return high_weight_indices[:estimated_cardinality]

        return sorted_indices[: min(estimated_cardinality, len(sorted_indices))]

    def get_number_of_targets(self):
        return len(self._get_extraction_indices())

    def get_point_estimate(self, flatten_vector=False):
        extraction_indices = self._get_extraction_indices()
        if not extraction_indices:
            if self.dim > 0:
                point_estimate = zeros((self.dim, 0))
            else:
                point_estimate = array([])
        else:
            point_estimate = column_stack(
                [self._components[index].mu for index in extraction_indices]
            )

        if flatten_vector:
            point_estimate = point_estimate.reshape((-1,))

        return point_estimate

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _gaussian_likelihood(innovation, covariance):
        covariance = GaussianMixturePHDFilter._symmetrize(covariance)
        determinant = max(float(linalg.det(covariance)), 1e-12)
        mahalanobis_distance = float(
            dot(innovation, linalg.solve(covariance, innovation))
        )
        normalization = (2.0 * pi) ** (innovation.shape[0] / 2.0) * determinant**0.5
        return float(exp(-0.5 * mahalanobis_distance) / normalization)

    @staticmethod
    def _get_measurement_covariance(cov_mat_meas, measurement_index):
        if cov_mat_meas.ndim == 2:
            return cov_mat_meas
        return cov_mat_meas[:, :, measurement_index]

    @staticmethod
    def _get_clutter_intensity(clutter_intensity, measurement_index):
        clutter_as_array = array(clutter_intensity)
        if clutter_as_array.ndim == 0 or clutter_as_array.size == 1:
            return float(clutter_as_array.reshape((-1,))[0])
        return float(clutter_as_array.reshape((-1,))[measurement_index])

    def _resolve_birth_arguments(self, birth_components, birth_weights):
        if birth_components is None:
            return self.birth_components, self.birth_weights
        if birth_weights is None:
            birth_weights = 0.05 * ones((len(birth_components),))
        birth_state = GaussianMixturePHDState(birth_components, birth_weights)
        return birth_state.dists, birth_state.w

    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    def predict_linear(
        self,
        system_matrix,
        sys_noise,
        inputs=None,
        birth_components=None,
        birth_weights=None,
        survival_probability=None,
    ):
        if isinstance(sys_noise, GaussianDistribution):
            assert all(sys_noise.mu == 0)
            sys_noise_cov = sys_noise.C
        else:
            sys_noise_cov = sys_noise

        if survival_probability is None:
            survival_probability = self.survival_probability

        predicted_components = []
        predicted_weights = []

        for weight, component in zip(self._weights, self._components):
            predicted_mean = system_matrix @ component.mu
            if inputs is not None:
                predicted_mean = predicted_mean + inputs

            predicted_covariance = (
                system_matrix @ component.C @ system_matrix.T + sys_noise_cov
            )
            predicted_covariance = self._symmetrize(predicted_covariance)

            predicted_components.append(
                GaussianDistribution(
                    predicted_mean,
                    predicted_covariance,
                    check_validity=False,
                )
            )
            predicted_weights.append(float(survival_probability * weight))

        birth_components_resolved, birth_weights_resolved = (
            self._resolve_birth_arguments(birth_components, birth_weights)
        )
        predicted_components.extend(copy.deepcopy(birth_components_resolved))

        if predicted_weights or birth_weights_resolved.size != 0:
            predicted_weights = hstack(
                (array(predicted_weights), birth_weights_resolved)
            )
        else:
            predicted_weights = zeros((0,))

        self._components = predicted_components
        self._weights = predicted_weights
        self._dim = system_matrix.shape[0]

        if self.log_prior_estimates:
            self.store_prior_estimates()

    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-statements
    def update_linear(
        self,
        measurements,
        measurement_matrix,
        cov_mat_meas,
        detection_probability=None,
        clutter_intensity=None,
    ):
        if detection_probability is None:
            detection_probability = self.detection_probability
        if clutter_intensity is None:
            clutter_intensity = self.clutter_intensity

        measurements = array(measurements)
        if measurements.ndim == 1:
            measurements = measurements.reshape((-1, 1))

        if len(self._components) == 0:
            if self.log_posterior_estimates:
                self.store_posterior_estimates()
            return

        if measurements.size == 0 or measurements.shape[1] == 0:
            self._weights = (1.0 - detection_probability) * self._weights
            self.prune()
            self.merge()
            self.cap_components()
            if self.log_posterior_estimates:
                self.store_posterior_estimates()
            return

        missed_detection_components = copy.deepcopy(self._components)
        missed_detection_weights = (1.0 - detection_probability) * self._weights

        updated_components = []
        updated_weights = []

        for measurement_index in range(measurements.shape[1]):
            measurement = measurements[:, measurement_index]
            curr_meas_cov = self._get_measurement_covariance(
                cov_mat_meas, measurement_index
            )

            measurement_components = []
            measurement_weights = []

            for component_index, component in enumerate(self._components):
                innovation_covariance = (
                    measurement_matrix @ component.C @ measurement_matrix.T
                    + curr_meas_cov
                )
                innovation_covariance = self._symmetrize(innovation_covariance)

                kalman_gain = linalg.solve(
                    innovation_covariance.T,
                    (measurement_matrix @ component.C).T,
                ).T
                innovation = measurement - measurement_matrix @ component.mu

                posterior_mean = component.mu + kalman_gain @ innovation
                posterior_covariance = (
                    component.C - kalman_gain @ innovation_covariance @ kalman_gain.T
                )
                posterior_covariance = self._symmetrize(posterior_covariance)

                likelihood = self._gaussian_likelihood(
                    innovation, innovation_covariance
                )
                weight = (
                    detection_probability * self._weights[component_index] * likelihood
                )

                measurement_components.append(
                    GaussianDistribution(
                        posterior_mean,
                        posterior_covariance,
                        check_validity=False,
                    )
                )
                measurement_weights.append(float(weight))

            if measurement_weights:
                normalization = self._get_clutter_intensity(
                    clutter_intensity, measurement_index
                ) + float(sum(array(measurement_weights)))
                if normalization <= 0.0:
                    continue

                updated_weights.extend(
                    [weight / normalization for weight in measurement_weights]
                )
                updated_components.extend(measurement_components)

        self._components = missed_detection_components + updated_components
        if updated_weights or missed_detection_weights.size != 0:
            self._weights = hstack((missed_detection_weights, array(updated_weights)))
        else:
            self._weights = zeros((0,))

        self.prune()
        self.merge()
        self.cap_components()

        if self.log_posterior_estimates:
            self.store_posterior_estimates()

    def prune(self, threshold=None):
        if threshold is None:
            threshold = self.pruning_threshold

        keep_indices = [
            index for index, weight in enumerate(self._weights) if weight > threshold
        ]
        self._components = [
            copy.deepcopy(self._components[index]) for index in keep_indices
        ]
        self._weights = (
            array([self._weights[index] for index in keep_indices])
            if keep_indices
            else zeros((0,))
        )

    def merge(self, threshold=None):
        if threshold is None:
            threshold = self.merging_threshold

        if len(self._components) <= 1:
            return

        remaining_components = copy.deepcopy(self._components)
        remaining_weights = [float(weight) for weight in self._weights]

        merged_components = []
        merged_weights = []

        while remaining_components:
            anchor_index = max(
                range(len(remaining_weights)), key=remaining_weights.__getitem__
            )
            anchor_component = remaining_components[anchor_index]

            cluster_indices = []
            for component_index, component in enumerate(remaining_components):
                difference = component.mu - anchor_component.mu
                distance = float(
                    dot(difference, linalg.solve(anchor_component.C, difference))
                )
                if distance <= threshold:
                    cluster_indices.append(component_index)

            cluster_weights = array(
                [remaining_weights[index] for index in cluster_indices]
            )
            total_weight = float(sum(cluster_weights))

            if total_weight <= 0.0:
                merged_components.append(copy.deepcopy(anchor_component))
                merged_weights.append(0.0)
            else:
                cluster_means = stack(
                    [remaining_components[index].mu for index in cluster_indices],
                    axis=0,
                )
                merged_mean = (
                    sum(cluster_means * cluster_weights[:, None], axis=0) / total_weight
                )

                merged_covariance = zeros_like(anchor_component.C)
                for local_index, component_index in enumerate(cluster_indices):
                    component = remaining_components[component_index]
                    difference = component.mu - merged_mean
                    merged_covariance = merged_covariance + cluster_weights[
                        local_index
                    ] * (component.C + outer(difference, difference))

                merged_covariance = self._symmetrize(merged_covariance / total_weight)
                merged_covariance = merged_covariance + 1e-9 * eye(
                    merged_covariance.shape[0]
                )

                merged_components.append(
                    GaussianDistribution(
                        merged_mean,
                        merged_covariance,
                        check_validity=False,
                    )
                )
                merged_weights.append(total_weight)

            cluster_index_set = set(cluster_indices)
            remaining_components = [
                component
                for index, component in enumerate(remaining_components)
                if index not in cluster_index_set
            ]
            remaining_weights = [
                weight
                for index, weight in enumerate(remaining_weights)
                if index not in cluster_index_set
            ]

        self._components = merged_components
        self._weights = array(merged_weights)

    def cap_components(self, max_components=None):
        if max_components is None:
            max_components = self.max_components

        if len(self._components) <= max_components:
            return

        keep_indices = list(argsort(self._weights)[::-1][:max_components])
        self._components = [
            copy.deepcopy(self._components[index]) for index in keep_indices
        ]
        self._weights = array([self._weights[index] for index in keep_indices])
