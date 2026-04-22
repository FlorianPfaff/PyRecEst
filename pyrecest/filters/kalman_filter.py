# pylint: disable=no-name-in-module,no-member

from pyrecest.backend import atleast_1d, atleast_2d, eye
from pyrecest.distributions import GaussianDistribution

from ._linear_gaussian import linear_gaussian_predict, linear_gaussian_update
from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin


class KalmanFilter(AbstractFilter, EuclideanFilterMixin):
    def __init__(self, initial_state):
        EuclideanFilterMixin.__init__(self)
        AbstractFilter.__init__(self, self._coerce_state(initial_state))

    @staticmethod
    def _coerce_state(state):
        if isinstance(state, GaussianDistribution):
            return GaussianDistribution(
                atleast_1d(state.mu),
                atleast_2d(state.C),
                check_validity=False,
            )
        if isinstance(state, tuple) and len(state) == 2:
            mean, covariance = state
            return GaussianDistribution(
                atleast_1d(mean),
                atleast_2d(covariance),
                check_validity=False,
            )
        raise ValueError(
            "state must be a GaussianDistribution or a tuple of (mean, covariance)"
        )

    @property
    def dim(self):
        """Returns the dimension of the state."""
        return self._filter_state.dim

    @property
    def filter_state(self) -> GaussianDistribution:
        return GaussianDistribution(
            self._filter_state.mu,
            self._filter_state.C,
            check_validity=False,
        )

    @filter_state.setter
    def filter_state(self, new_state):
        """
        Set the filter state.

        :param new_state: Provide GaussianDistribution or mean and covariance as state.
        """
        self._filter_state = self._coerce_state(new_state)

    def predict_identity(self, sys_noise_cov, sys_input=None):
        """
        Predicts the next state assuming identity transition matrix.

        :param sys_noise_cov: System noise covariance.
        :param sys_input: System input.
        """
        self.predict_linear(eye(self.dim), sys_noise_cov, sys_input)

    def predict_linear(
        self,
        system_matrix,
        sys_noise_cov,
        sys_input=None,
    ):
        """
        Predicts the next state assuming a linear system model.

        :param system_matrix: System transition matrix.
        :param sys_noise_cov: System noise covariance.
        :param sys_input: System input.
        """
        new_mean, new_covariance = linear_gaussian_predict(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            system_matrix=system_matrix,
            sys_noise_cov=sys_noise_cov,
            sys_input=sys_input,
        )
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )

    def update_identity(self, meas_noise, measurement):
        """
        Update the filter state with measurement, assuming identity measurement matrix.

        :param measurement: Measurement.
        :param meas_noise: Measurement noise covariance.
        """
        self.update_linear(
            measurement=measurement,
            measurement_matrix=eye(self.dim),
            meas_noise=meas_noise,
        )

    def update_linear(
        self,
        measurement,
        measurement_matrix,
        meas_noise,
    ):
        """
        Update the filter state with measurement, assuming a linear measurement model.

        :param measurement: Measurement.
        :param measurement_matrix: Measurement matrix.
        :param meas_noise: Covariance matrix for measurement.
        """
        new_mean, new_covariance = linear_gaussian_update(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            measurement=measurement,
            measurement_matrix=measurement_matrix,
            meas_noise=meas_noise,
        )
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )

    def get_point_estimate(self):
        """Returns the mean of the current filter state."""
        return self._filter_state.mu
