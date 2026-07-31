import numpy as np

from pyrecest.utils.metrics import gospa_distance, ospa_distance


def _large_coordinate_case():
    component = 1.0e308
    estimated = np.array([[component, component]])
    reference = np.zeros((1, 2))
    expected = np.hypot(component, component)
    cutoff = 1.5e308
    return estimated, reference, expected, cutoff


def test_ospa_preserves_representable_large_coordinate_distance():
    estimated, reference, expected, cutoff = _large_coordinate_case()

    with np.errstate(over="raise", invalid="raise"):
        distance = ospa_distance(
            estimated,
            reference,
            cutoff=cutoff,
            order=1.0,
        )

    assert np.isfinite(distance)
    np.testing.assert_allclose(
        distance,
        expected,
        rtol=4.0 * np.finfo(float).eps,
        atol=0.0,
    )


def test_gospa_preserves_representable_large_coordinate_distance():
    estimated, reference, expected, cutoff = _large_coordinate_case()

    with np.errstate(over="raise", invalid="raise"):
        distance = gospa_distance(
            estimated,
            reference,
            cutoff=cutoff,
            order=1.0,
        )

    assert np.isfinite(distance)
    np.testing.assert_allclose(
        distance,
        expected,
        rtol=4.0 * np.finfo(float).eps,
        atol=0.0,
    )
