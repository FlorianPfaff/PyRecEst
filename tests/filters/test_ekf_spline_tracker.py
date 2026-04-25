import unittest

from tests.filters.spline_tracker_test_cases import SplineTrackerCommonTests


class TestEKFSplineTracker(SplineTrackerCommonTests, unittest.TestCase):
    tracker_key = "ekf"
