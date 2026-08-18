from pyrecest.backend import array
from pyrecest.distributions import HypertoroidalDiracDistribution


def test_apply_function_updates_dimension_after_coordinate_reduction():
    distribution = HypertoroidalDiracDistribution(
        array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        ),
        array([0.2, 0.3, 0.5]),
    )

    transformed = distribution.apply_function(lambda points: points[:, :2])

    assert distribution.dim == 3
    assert transformed.dim == 2
    assert transformed.d.shape == (3, 2)
    assert transformed.w.shape == (3,)
    assert transformed.trigonometric_moment(1).shape == (2,)
