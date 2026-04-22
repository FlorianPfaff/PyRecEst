import copy
import warnings
from abc import abstractmethod

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import dstack, ndim, stack
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

    @abstractmethod
    def find_association(self, measurements, measurement_matrix, cov_mats_meas):
        """
        This method must be implemented in subclass
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_number_of_targets(self) -> int:
        return len(self.filter_bank)

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
            assert all(
                id(new_state[i]) != id(new_state[j])
                for i in range(len(new_state))
                for j in range(i + 1, len(new_state))
            ), "No two filters of the filter bank should have the same handle. Updating the state of one target would update it for all!"
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

        assert all(
            dim == self.filter_bank[0].dim for dim in system_matrices.shape[:2]
        ), "system_matrices may be a single (dimSingleState, dimSingleState) matrix or a (dimSingleState, dimSingleState, noTargets) tensor."

        if isinstance(sys_noises, GaussianDistribution):
            assert all(sys_noises.mu == 0)
            sys_noises = dstack(sys_noises.C)

        curr_sys_matrix = system_matrices
        curr_sys_noise = sys_noises
        curr_input = inputs

        for i in range(self.get_number_of_targets()):
            if system_matrices is not None and ndim(system_matrices) == 3:
                curr_sys_matrix = system_matrices[:, :, i]
            if sys_noises is not None and ndim(sys_noises) == 3:
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
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"
        if len(self.filter_bank) == 0:
            warnings.warn("Currently, there are zero targets")
            return
        assert (
            measurement_matrix.shape[0] == measurements.shape[0]
            and measurement_matrix.shape[1]
            == self.filter_bank[0].get_point_estimate().shape[0]
        ), "Dimensions of measurement matrix must match state and measurement dimensions."

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
