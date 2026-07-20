import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import to_numpy
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)


def _as_float(value):
    return float(np.asarray(to_numpy(value)))


@pytest.mark.parametrize("moment", [0.5 + 0.5j, [0.5 + 0.5j]])
def test_from_moment_accepts_scalar_and_singleton_array_like_inputs(moment):
    dist = WrappedNormalDistribution.from_moment(moment)
    expected_moment = 0.5 + 0.5j

    npt.assert_allclose(_as_float(dist.scalar_mu), np.pi / 4.0, atol=1e-7)
    npt.assert_allclose(
        _as_float(dist.sigma),
        np.sqrt(-2.0 * np.log(abs(expected_moment))),
        atol=1e-7,
    )


def test_from_moment_rejects_vector_moments_with_clear_error():
    with pytest.raises(ValueError, match="First trigonometric moment must be a scalar"):
        WrappedNormalDistribution.from_moment([0.25 + 0.0j, 0.5 + 0.0j])


def test_from_moment_rejects_zero_moment_with_clear_error():
    with pytest.raises(
        ValueError,
        match="zero first trigonometric moment cannot be represented",
    ):
        WrappedNormalDistribution.from_moment(0.0)
