from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import pyrecest.distributions.ellipsoidal_ball_uniform_distribution as ellipsoid_module
from pyrecest.backend import array, eye
from pyrecest.distributions import EllipsoidalBallUniformDistribution


def test_sampling_handles_zero_length_gaussian_direction():
    distribution = EllipsoidalBallUniformDistribution(array([0.0, 0.0]), eye(2))
    gaussian_draws = array([[0.0, 0.0], [0.0, 2.0]])
    radial_draws = array([[0.25], [0.25]])

    with (
        patch.object(ellipsoid_module.random, "normal", return_value=gaussian_draws),
        patch.object(ellipsoid_module.random, "uniform", return_value=radial_draws),
        np.errstate(divide="raise", invalid="raise"),
    ):
        samples = distribution.sample(2)

    npt.assert_allclose(samples, array([[0.5, 0.0], [0.0, 0.5]]))
    assert np.all(np.isfinite(np.asarray(samples)))
