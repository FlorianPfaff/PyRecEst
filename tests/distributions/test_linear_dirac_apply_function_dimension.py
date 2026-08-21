import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution


class _LinearDiracSubclass(LinearDiracDistribution):
    pass


def test_apply_function_rebuilds_dimension_after_shape_change():
    dist = LinearDiracDistribution(
        array([[0.0, 1.0], [2.0, 3.0]]),
        array([0.25, 0.75]),
    )

    mapped = dist.apply_function(lambda particles: particles[:, :1])

    assert dist.dim == 2
    assert mapped.dim == 1
    assert tuple(mapped.d.shape) == (2, 1)
    assert tuple(mapped.mean().shape) == (1,)
    assert tuple(mapped.covariance().shape) == (1, 1)
    npt.assert_allclose(mapped.d, array([[0.0], [2.0]]))
    npt.assert_allclose(mapped.w, dist.w)


def test_apply_function_preserves_linear_dirac_subclass():
    dist = _LinearDiracSubclass(
        array([[0.0, 1.0], [2.0, 3.0]]),
        array([0.25, 0.75]),
    )

    mapped = dist.apply_function(lambda particles: particles[:, :1])

    assert type(mapped) is _LinearDiracSubclass
    assert mapped.dim == 1
    npt.assert_allclose(mapped.d, array([[0.0], [2.0]]))
    npt.assert_allclose(mapped.w, dist.w)
