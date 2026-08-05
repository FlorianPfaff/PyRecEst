import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import array
from pyrecest.distributions import (
    CircularDiracDistribution,
    CircularUniformDistribution,
    HypertoroidalDiracDistribution,
)


def _distribution():
    return HypertoroidalDiracDistribution(
        array([[0.1, 0.2], [0.3, 0.4]]),
        array([0.25, 0.75]),
    )


def test_dimension_indices_accept_zero_dimensional_integer_arrays():
    dist = _distribution()

    marginal = dist.marginalize_to_1D(np.asarray(1, dtype=np.int64))
    marginalized_out = dist.marginalize_out(np.asarray(0, dtype=np.int64))

    assert isinstance(marginal, CircularDiracDistribution)
    assert isinstance(marginalized_out, CircularDiracDistribution)
    npt.assert_allclose(marginal.d, dist.d[:, 1])
    npt.assert_allclose(marginalized_out.d, dist.d[:, 1])


def test_particle_count_accepts_zero_dimensional_integer_array():
    approximation = HypertoroidalDiracDistribution.from_distribution(
        CircularUniformDistribution(),
        np.asarray(3, dtype=np.int64),
    )

    assert approximation.d.shape[0] == 3
    assert approximation.w.shape == (3,)


@pytest.mark.parametrize(
    "operation",
    [
        lambda dist, value: dist.trigonometric_moment(value),
        lambda dist, value: dist.marginalize_to_1D(value),
        lambda dist, value: dist.marginalize_out(value),
    ],
)
def test_masked_integer_scalars_are_rejected(operation):
    with pytest.raises(ValueError, match="integer|dimension"):
        operation(_distribution(), np.ma.array(1, mask=True))
