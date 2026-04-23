from abc import abstractmethod

import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import vstack

from .abstract_tracker_with_logging import AbstractTrackerWithLogging


# pylint: disable=too-many-instance-attributes
class AbstractExtendedObjectTracker(AbstractTrackerWithLogging):
    def __init__(
        self,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )
        self.log_prior_extents = log_prior_extents
        self.log_posterior_extents = log_posterior_extents
        if log_prior_extents:
            self.prior_extents_over_time = self.history.register(
                "prior_extents", pad_with_nan=True
            )
        if log_posterior_extents:
            self.posterior_extents_over_time = self.history.register(
                "posterior_extents", pad_with_nan=True
            )

    def store_prior_estimates(self):
        curr_ests = self.get_point_estimate()
        # pylint: disable=W0201
        self.prior_estimates_over_time = self._record_estimates(
            "prior_estimates", curr_ests
        )

    def store_posterior_estimates(self):
        curr_ests = self.get_point_estimate()
        # pylint: disable=W0201
        self.posterior_estimates_over_time = self._record_estimates(
            "posterior_estimates", curr_ests
        )

    def store_prior_extent(self):
        curr_ext = self.get_point_estimate_extent()
        # pylint: disable=W0201
        self.prior_extents_over_time = self._record_estimates("prior_extents", curr_ext)

    def store_posterior_extents(self):
        curr_ext = self.get_point_estimate_extent()
        # pylint: disable=W0201
        self.posterior_extents_over_time = self._record_estimates(
            "posterior_extents", curr_ext
        )

    @abstractmethod
    def get_point_estimate(self):
        """
        Retrieve the estimated kinematic state and extent of the object,
        all flattened into a single vector.

        Returns:
        - A vector representing both the estimated kinematic state and extent.
        """

    @abstractmethod
    def get_point_estimate_kinematics(self):
        """
        Retrieve the estimated kinematic state of the object.

        Returns:
        - A vector representing the estimated kinematic state.
        """

    @abstractmethod
    def get_point_estimate_extent(self, flatten_matrix=False):
        """
        Retrieve the estimated extent of the object.

        Parameters:
        - flatten_matrix: whether to flatten the extent matrix or not

        Returns:
        - A matrix or vector representing the estimated extent.
        """

    @abstractmethod
    def get_contour_points(self, n):
        pass

    def plot_point_estimate(self):
        contour = self.get_contour_points(100)
        closed_contour = vstack((contour, contour[-1]))
        plt.plot(closed_contour[:, 0], closed_contour[:, 1])
