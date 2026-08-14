import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from pyrecest.backend import to_numpy
from pyrecest.sampling.hyperspherical_sampler import DriscollHealySampler


class _FakeGrid:
    nlat = 3
    nlon = 2

    @staticmethod
    def lons():
        return np.array([0.0, 90.0])

    @staticmethod
    def lats():
        return np.array([90.0, 0.0, -45.0])


class _FakeSHGrid:
    @staticmethod
    def from_zeros(grid_density_parameter):
        assert grid_density_parameter == 2
        return _FakeGrid()


def test_driscoll_healy_latitudes_map_to_correct_cartesian_rows():
    fake_pyshtools = ModuleType("pyshtools")
    fake_pyshtools.SHGrid = _FakeSHGrid

    with patch.dict(sys.modules, {"pyshtools": fake_pyshtools}):
        samples, description = DriscollHealySampler().get_grid(2)

    samples = np.asarray(to_numpy(samples), dtype=float)
    root_half = np.sqrt(0.5)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [root_half, 0.0, -root_half],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, root_half, -root_half],
        ]
    )

    npt.assert_allclose(samples, expected, atol=1e-12)
    assert description == {
        "scheme": "driscoll_healy",
        "l_max": 2,
        "n_lat": 3,
        "n_lon": 2,
    }
