import copy
import warnings
from abc import abstractmethod

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import any as backend_any
from pyrecest.backend import ndim, stack
from pyrecest.distributions import GaussianDistribution

from .abstract_multitarget_tracker import AbstractMultitargetTracker
from .kalman_filter import KalmanFilter
from .manifold_mixins import EuclideanFilterMixin


class AbstractNearestNeighborTracker(AbstractMultitargetTracker):
    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        AbstractMultitargetTracker.__init__(
            self, log_prior_estimates, log_posterior_estimates
        )
        self.association_param = association_param or {}

        if initial_prior is not None:
            self.filter_state = initial_prior
        else:
            self._filter_state = None

    @staticmethod
    def _ensure_numpy_backend():
        if pyrecest.backend.__backend_name__ != "numpy":
            raise NotImplementedError(
                "Nearest-neighbor trackers are only supported for the numpy backend."
            )

    @staticmethod
    def _validate_unique_filter_handles(new_state):
        if any(
            id(new_state[i]) == id(new_state[j])
            for i in range(len(new_state))
            for j in range(i + 1, len(new_state))
        ):
            raise ValueError(
                "No two filters of the filter bank should have the same handle. "
                "Updating the state of one target would update it for all!"
            )

    def _validate_measurement_matrix_shape(self, measurement_matrix, measurements):
        if (
            measurement_matrix.shape[0] != measurements.shape[0]
            or measurement_matrix.shape[1]
            != self.filter_bank[0].get_point_estimate().shape[0]
        ):
            raise ValueError(
                "Dimensions of measurement matrix must match state and measurement dimensions."
            )

    @abstractmethod
    def find_association(self, measurements, measurement_matrix, cov_mats_meas):
        """
        This method must be implemented in subclass
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_number_of_targets(self) -> int:
        return len(self.filter_bank)

    @staticmethod
    def _require_numpy_backend(operation: str):
        if pyrecest.backend.__backend_name__ != "numpy":
            raise NotImplementedError(
                f"{operation} is only supported for the numpy backend"
            )

    @staticmethod
    def _prediction_input_shape(value):
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = pyrecest.backend.asarray(value).shape
        return tuple(shape)

    def _validate_prediction_input_shapes(
        self, system_matrices, sys_noises, inputs
    ):
        n_targets = self.get_number_of_targets()
        state_dim = self.filter_bank[0].dim

        if any(filter_obj.dim != state_dim for filter_obj in self.filter_bank[1:]):
            raise ValueError("All target filters must have the same state dimension.")

        shared_matrix_shapes = {(state_dim, state_dim)}
        shared_input_shapes = {(state_dim,)}
        if state_dim == 1:
            # Preserve the scalar conveniences accepted by the Kalman primitive.
            shared_matrix_shapes.update({(), (1,)})
            shared_input_shapes.add(())

        system_matrix_shape = self._prediction_input_shape(system_matrices)
        valid_system_matrix_shapes = shared_matrix_shapes | {
            (state_dim, state_dim, n_targets)
        }
        if system_matrix_shape not in valid_system_matrix_shapes:
            raise ValueError(
                "system_matrices may be a single (dimSingleState, dimSingleState) "
                "matrix or a (dimSingleState, dimSingleState, noTargets) tensor."
            )

        sys_noise_shape = self._prediction_input_shape(sys_noises)
        valid_sys_noise_shapes = shared_matrix_shapes | {
            (state_dim, state_dim, n_targets)
        }
        if sys_noise_shape not in valid_sys_noise_shapes:
            raise ValueError(
                "sys_noises may be a single (dimSingleState, dimSingleState) "
                "matrix or a (dimSingleState, dimSingleState, noTargets) tensor."
            )

        if inputs is not None:
            input_shape = self._prediction_input_shape(inputs)
            valid_input_shapes = shared_input_shapes | {(state_dim, n_targets)}
            if input_shape not in valid_input_shapes:
                raise ValueError(
                    "inputs may be a single (dimSingleState,) vector or a "
                    "(dimSingleState, noTargets) matrix."
                )

    @staticmethod
    def _validate_measurement_update_inputs(
        measurements, measurement_matrix, state_dim
    ):
        if measurements.ndim != 2:
            raise ValueError("measurements must have shape (dim_meas, n_meas).")
        if measurement_matrix.ndim != 2:
            raise ValueError("measurement_matrix must be a 2D matrix.")
        if (
            measurement_matrix.shape[0] != measurements.shape[0]
            or measurement_matrix.shape[1] != state_dim
        ):
            raise ValueError(
                "Dimensions of measurement matrix must match state and measurement dimensions."
            )

    @property
    def dim(self) -> int:
        if not self.filter_bank:
            raise ValueError("Cannot provide state dimension if there are no targets.")
        return self.filter_bank[0].dim

    @property
    def filter_state(self):
        if self.get_number_of_targets() == 0:
            warnings.warn("Currently, there are zero targets.")
            return None

        dists = [self.filter_bank[0].filter_state]
        for i in range(1, self.get_number_of_targets()):
            dists.append(self.filter_bank[i].filter_state)
        return dists

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, list) and all(
            isinstance(item, EuclideanFilterMixin) for item in new_state
        ):
            self._validate_unique_filter_handles(new_state)
            self.filter_bank = copy.deepcopy(new_state)
        else:
            self.filter_bank = [
                KalmanFilter(filter_state) for filter_state in new_state
            ]

        if self.log_prior_estimates:
            self.store_prior_estimates()

    def predict_linear(self, system_matrices, sys_noises, inputs=None):
        if not self.filter_bank:
            warnings.warn("Currently, there are zero targets.")
            return

        if isinstance(sys_noises, GaussianDistribution):
            if bool(backend_any(sys_noises.mu != 0)):
                raise ValueError("Gaussian process noise must have zero mean.")
            sys_noises = sys_noises.C

        self._validate_prediction_input_shapes(system_matrices, sys_noises, inputs)

        curr_sys_matrix = system_matrices
        curr_sys_noise = sys_noises
        curr_input = inputs

        for i in range(self.get_number_of_targets()):
            if ndim(system_matrices) == 3:
                curr_sys_matrix = system_matrices[:, :, i]
            if ndim(sys_noises) == 3:
                curr_sys_noise = sys_noises[:, :, i]
            if inputs is not None and ndim(inputs) == 2:
                curr_input = inputs[:, i]

            self.filter_bank[i].predict_linear(
                curr_sys_matrix, curr_sys_noise, curr_input
            )

        if self.log_prior_estimates:
            self.store_prior_estimates()

    def update_linear(
        self,
        measurements,
        measurement_matrix,
        covMatsMeas,
        pairwise_cost_matrix=None,
    ):
        self._require_numpy_backend("update_linear")
        if len(self.filter_bank) == 0:
            warnings.warn("Currently, there are zero targets")
            return
        self._validate_measurement_update_inputs(
            measurements,
            measurement_matrix,
            self.filter_bank[0].get_point_estimate().shape[0],
        )

        if pairwise_cost_matrix is None:
            association = self.find_association(
                measurements, measurement_matrix, covMatsMeas
            )
        else:
            if not hasattr(self, "find_association_from_cost_matrix"):
                raise NotImplementedError(
                    "This tracker does not support pairwise_cost_matrix."
                )
            association = self.find_association_from_cost_matrix(pairwise_cost_matrix)

        currMeasCov = covMatsMeas
        for i in range(self.get_number_of_targets()):
            if association[i] < measurements.shape[1]:
                if covMatsMeas.ndim != 2:
                    currMeasCov = covMatsMeas[:, :, association[i]]
                self.filter_bank[i].update_linear(
                    measurements[:, association[i]], measurement_matrix, currMeasCov
                )
        if self.log_posterior_estimates:
            self.store_posterior_estimates()

    def get_point_estimate(self, flatten_vector=False):
        num_targets = self.get_number_of_targets()
        if num_targets == 0:
            warnings.warn("Currently, there are zero targets.")
            point_ests = None
        else:
            point_ests = stack(
                [filter_obj.get_point_estimate() for filter_obj in self.filter_bank],
                axis=1,
            )
            if flatten_vector:
                point_ests = point_ests.flatten()
        return point_ests
