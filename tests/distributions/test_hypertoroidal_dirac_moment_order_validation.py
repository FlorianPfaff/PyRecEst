import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions import HypertoroidalDiracDistribution


def _distribution():
    return HypertoroidalDiracDistribution(
        array([[0.1, 0.2], [0.3, 0.4]]),
        array([0.25, 0.75]),
    )


@pytest.mark.parametrize(
    "invalid_order",
    [True, False, 1.5, "1", np.array([1]), np.asarray(1.0)],
)
def test_trigonometric_moment_rejects_noninteger_orders(invalid_order):
    with pytest.raises(ValueError, match="integer"):
        _distribution().trigonometric_moment(invalid_order)


def test_trigonometric_moment_accepts_integer_scalar_types():
    dist = _distribution()
    expected = pyrecest.backend.to_numpy(dist.trigonometric_moment(2))

    for order in (np.int64(2), np.asarray(2, dtype=np.int64)):
        actual = pyrecest.backend.to_numpy(dist.trigonometric_moment(order))
        npt.assert_allclose(actual, expected)


def test_trigonometric_moment_preserves_negative_orders():
    dist = _distribution()
    positive = pyrecest.backend.to_numpy(dist.trigonometric_moment(1))
    negative = pyrecest.backend.to_numpy(dist.trigonometric_moment(-1))

    npt.assert_allclose(negative, np.conjugate(positive))
