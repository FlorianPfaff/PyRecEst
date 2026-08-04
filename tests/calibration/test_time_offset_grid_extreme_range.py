import numpy as np
import numpy.testing as npt

from pyrecest.calibration.time_offset import make_offset_grid


def test_make_offset_grid_avoids_overflow_for_extreme_finite_span():
    max_float = np.finfo(float).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        offsets = make_offset_grid(-max_float, max_float, max_float)

    npt.assert_array_equal(offsets, np.array([-max_float, 0.0, max_float]))
