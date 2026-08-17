"""Regression tests for stable SO(3) product quaternion normalization."""

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import SO3ProductDiracDistribution


def test_product_distribution_normalizes_extreme_finite_scales():
    backend_dtype = to_numpy(array([1.0])).dtype
    finfo = np.finfo(backend_dtype)
    locations = array(
        np.asarray(
            [
                [[finfo.max, finfo.max / 2.0, 0.0, 0.0]],
                [[2.0 * finfo.tiny, finfo.tiny, 0.0, 0.0]],
            ],
            dtype=backend_dtype,
        )
    )
    expected_direction = np.asarray(
        [2.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 0.0, 0.0],
        dtype=backend_dtype,
    )

    distribution = SO3ProductDiracDistribution(locations)
    normalized = to_numpy(distribution.d[:, 0, :])

    assert np.all(np.isfinite(normalized))
    npt.assert_allclose(
        normalized,
        np.stack([expected_direction, expected_direction]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_geodesic_distance_preserves_scalar_result_for_single_rotations():
    rotation = array([0.0, 0.0, 0.0, 1.0])

    distance = SO3ProductDiracDistribution.geodesic_distance(rotation, rotation)

    assert np.isscalar(distance)
    assert float(distance) == 0.0
