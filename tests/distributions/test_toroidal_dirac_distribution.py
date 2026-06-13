import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import ToroidalDiracDistribution


def _as_numpy(value):
    return np.asarray(pyrecest.backend.to_numpy(value))


def _make_toroidal_dirac_distribution():
    return ToroidalDiracDistribution(
        array(
            [
                [0.1, 0.2],
                [1.0, 2.0],
                [2.5, 0.4],
                [4.0, 5.0],
            ]
        ),
        array([0.1, 0.2, 0.3, 0.4]),
    )


class TestToroidalDiracDistribution(unittest.TestCase):
    def test_circular_correlation_uses_particle_rows(self):
        dist = _make_toroidal_dirac_distribution()
        d = _as_numpy(dist.d)
        w = _as_numpy(dist.w)
        mean = _as_numpy(dist.mean_direction())

        first_sines = np.sin(d[:, 0] - mean[0])
        second_sines = np.sin(d[:, 1] - mean[1])
        expected = np.sum(w * first_sines * second_sines) / np.sqrt(
            np.sum(w * first_sines**2) * np.sum(w * second_sines**2)
        )

        npt.assert_allclose(
            _as_numpy(dist.circular_correlation_jammalamadaka()),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_covariance_4d_uses_particle_rows(self):
        dist = _make_toroidal_dirac_distribution()
        d = _as_numpy(dist.d)
        w = _as_numpy(dist.w)

        dbar = np.column_stack(
            [
                np.cos(d[:, 0]),
                np.sin(d[:, 0]),
                np.cos(d[:, 1]),
                np.sin(d[:, 1]),
            ]
        )
        mean = np.sum(w[:, None] * dbar, axis=0)
        centered = dbar - mean
        expected = centered.T @ (w[:, None] * centered)

        actual = _as_numpy(dist.covariance_4D())

        self.assertEqual(actual.shape, (4, 4))
        npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
