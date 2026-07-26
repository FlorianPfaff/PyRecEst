import numpy as np
import numpy.testing as npt
from pyrecest.calibration.time_offset import make_offset_grid


def test_preserves_subnanosecond_offset_steps():
    step_s = 2.5e-10
    offsets = make_offset_grid(0.0, 1.0e-9, step_s)
    expected = np.arange(5, dtype=float) * step_s

    npt.assert_array_equal(offsets, expected)
    assert np.unique(offsets).size == offsets.size
