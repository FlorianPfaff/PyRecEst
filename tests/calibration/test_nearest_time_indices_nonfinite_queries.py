import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.calibration.time_offset import nearest_time_indices


class NearestTimeIndicesNonfiniteQueriesTest(unittest.TestCase):
    def test_nonfinite_query_times_are_marked_unmatched(self):
        indices = nearest_time_indices(
            np.array([0.0, 2.0, np.nan]),
            np.array([np.nan, 0.4, np.inf, 1.7]),
        )

        npt.assert_array_equal(indices, np.array([-1, 0, -1, 1]))


if __name__ == "__main__":
    unittest.main()
