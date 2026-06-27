import unittest
from unittest.mock import patch

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye
from pyrecest.filters.abstract_extended_object_tracker import (
    AbstractExtendedObjectTracker,
)


class _MinimalExtendedObjectTracker(AbstractExtendedObjectTracker):
    def __init__(self):
        super().__init__()
        self.contour = array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )

    def get_point_estimate(self):
        return array([0.0, 0.0])

    def get_point_estimate_kinematics(self):
        return array([0.0, 0.0])

    def get_point_estimate_extent(self, flatten_matrix=False):
        extent = eye(2)
        if flatten_matrix:
            return extent.flatten()
        return extent

    def get_contour_points(self, n):
        return self.contour


class TestAbstractExtendedObjectTracker(unittest.TestCase):
    @patch("matplotlib.pyplot.plot")
    def test_plot_point_estimate_closes_contour_with_first_point(self, mock_plot):
        tracker = _MinimalExtendedObjectTracker()

        tracker.plot_point_estimate()

        mock_plot.assert_called_once()
        args, _ = mock_plot.call_args
        x_values, y_values = args
        npt.assert_array_equal(x_values, array([0.0, 1.0, 1.0, 0.0]))
        npt.assert_array_equal(y_values, array([0.0, 0.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
