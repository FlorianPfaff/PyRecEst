from abc import abstractmethod

from .abstract_tracker_with_logging import AbstractTrackerWithLogging


class AbstractMultitargetTracker(AbstractTrackerWithLogging):
    def __init__(self, log_prior_estimates=True, log_posterior_estimates=True):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

    def store_prior_estimates(self):
        curr_ests = self.get_point_estimate(True)
        # pylint: disable=W0201
        self.prior_estimates_over_time = self._record_estimates(
            "prior_estimates", curr_ests
        )

    def store_posterior_estimates(self):
        curr_ests = self.get_point_estimate(True)
        # pylint: disable=W0201
        self.posterior_estimates_over_time = self._record_estimates(
            "posterior_estimates", curr_ests
        )

    @abstractmethod
    def get_point_estimate(self, flatten_vector=False):
        pass

    @abstractmethod
    def get_number_of_targets(self):
        pass
