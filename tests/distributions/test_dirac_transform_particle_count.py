import pytest
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution


def make_distribution():
    return LinearDiracDistribution(
        array([[0.0, 1.0], [2.0, 3.0]]),
        array([0.25, 0.75]),
    )


def test_vectorized_transform_rejects_changed_particle_count():
    distribution = make_distribution()

    with pytest.raises(ValueError, match="preserve the number of Dirac locations"):
        distribution.apply_function(lambda points: points[:1])


def test_vectorized_transform_rejects_scalar_output():
    distribution = make_distribution()

    with pytest.raises(ValueError, match="preserve the number of Dirac locations"):
        distribution.apply_function(lambda points: 1.0)


def test_vectorized_transform_may_change_coordinate_dimension():
    distribution = make_distribution()

    transformed = distribution.apply_function(
        lambda points: points[:, :1],
        function_is_vectorized=True,
    )

    assert transformed.d.shape == (2, 1)
    assert transformed.w.shape == (2,)
    assert distribution.d.shape == (2, 2)
