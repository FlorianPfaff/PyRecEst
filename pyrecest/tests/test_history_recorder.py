import unittest
from math import nan

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, zeros

from pyrecest.filters.abstract_extended_object_tracker import AbstractExtendedObjectTracker
from pyrecest.filters.abstract_filter import AbstractFilter
from pyrecest.filters.abstract_multitarget_tracker import AbstractMultitargetTracker
from pyrecest.utils import HistoryRecorder


class _DummyBelief:
    def __init__(self, mean):
        self._mean = array(mean)
        self.dim = self._mean.size

    def mean(self):
        return self._mean

    def plot(self):
        return None


class _DummyFilter(AbstractFilter):
    def __init__(self, initial_filter_state=None):
        AbstractFilter.__init__(self, initial_filter_state)


class _DummyMultitargetTracker(AbstractMultitargetTracker):
    def __init__(self):
        super().__init__(log_prior_estimates=True, log_posterior_estimates=True)
        self.estimate = array([])

    def get_point_estimate(self, flatten_vector=False):
        if flatten_vector:
            return self.estimate
        return self.estimate.reshape(-1, 1)

    def get_number_of_targets(self):
        return 1


class _DummyExtendedTracker(AbstractExtendedObjectTracker):
    def __init__(self):
        super().__init__(
            log_prior_estimates=True,
            log_posterior_estimates=True,
            log_prior_extents=True,
            log_posterior_extents=True,
        )
        self.estimate = array([])
        self.extent = array([[]])

    def get_point_estimate(self):
        return self.estimate

    def get_point_estimate_kinematics(self):
        return self.estimate

    def get_point_estimate_extent(self, flatten_matrix=False):
        if flatten_matrix:
            return self.extent.flatten()
        return self.extent

    def get_contour_points(self, n):
        return zeros((n, 2))


class HistoryRecorderTest(unittest.TestCase):
    def test_padded_history_grows_and_preserves_old_columns(self):
        recorder = HistoryRecorder()
        recorder.register("estimate", pad_with_nan=True)

        recorder.record("estimate", array([1.0, 2.0]), pad_with_nan=True)
        recorder.record("estimate", array([3.0]), pad_with_nan=True)
        recorder.record("estimate", array([4.0, 5.0, 6.0]), pad_with_nan=True)

        expected = array(
            [[1.0, 3.0, 4.0], [2.0, nan, 5.0], [nan, nan, 6.0]]
        )
        npt.assert_allclose(recorder["estimate"], expected, equal_nan=True)

    def test_object_history_deep_copies_values(self):
        recorder = HistoryRecorder()
        state = {"x": [1, 2]}
        recorder.record("state", state)
        state["x"].append(3)

        self.assertEqual(recorder["state"][0], {"x": [1, 2]})

    def test_filter_helpers_record_state_and_point_estimate(self):
        filter_obj = _DummyFilter(_DummyBelief([1.0, 2.0]))
        filter_obj.record_filter_state()
        filter_obj.record_point_estimate()
        filter_obj.filter_state = _DummyBelief([3.0, 4.0])
        filter_obj.record_point_estimate()

        recorded_state = filter_obj.history["filter_state"][0]
        self.assertIsNot(recorded_state, filter_obj.filter_state)
        npt.assert_array_equal(recorded_state.mean(), array([1.0, 2.0]))
        npt.assert_array_equal(
            filter_obj.history["point_estimate"], array([[1.0, 3.0], [2.0, 4.0]])
        )

    def test_multitarget_tracker_logging_is_mirrored_in_history(self):
        tracker = _DummyMultitargetTracker()
        tracker.estimate = array([1.0, 2.0])
        tracker.store_prior_estimates()
        tracker.estimate = array([3.0])
        tracker.store_prior_estimates()
        tracker.estimate = array([4.0, 5.0, 6.0])
        tracker.store_posterior_estimates()

        expected_prior = array([[1.0, 3.0], [2.0, nan]])
        expected_posterior = array([[4.0], [5.0], [6.0]])
        npt.assert_allclose(tracker.prior_estimates_over_time, expected_prior, equal_nan=True)
        npt.assert_allclose(tracker.history["prior_estimates"], expected_prior, equal_nan=True)
        npt.assert_allclose(tracker.posterior_estimates_over_time, expected_posterior, equal_nan=True)
        npt.assert_allclose(tracker.history["posterior_estimates"], expected_posterior, equal_nan=True)

    def test_extended_tracker_logs_extents_via_same_recorder(self):
        tracker = _DummyExtendedTracker()
        tracker.estimate = array([1.0, 2.0])
        tracker.extent = array([[1.0, 0.0], [0.0, 1.0]])
        tracker.store_prior_estimates()
        tracker.store_prior_extent()

        tracker.estimate = array([3.0])
        tracker.extent = array([[2.0]])
        tracker.store_posterior_estimates()
        tracker.store_posterior_extents()

        expected_prior_extents = array([[1.0], [0.0], [0.0], [1.0]])
        expected_posterior_extents = array([[2.0]])
        npt.assert_allclose(
            tracker.history["prior_extents"], expected_prior_extents, equal_nan=True
        )
        npt.assert_allclose(
            tracker.history["posterior_extents"], expected_posterior_extents, equal_nan=True
        )
        npt.assert_allclose(
            tracker.prior_extents_over_time, expected_prior_extents, equal_nan=True
        )
        npt.assert_allclose(
            tracker.posterior_extents_over_time,
            expected_posterior_extents,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
