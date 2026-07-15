import math

import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.hypertorus.toroidal_von_mises_cosine_distribution import (
    ToroidalVonMisesCosineDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import (
    ToroidalVonMisesSineDistribution,
)
from scipy.special import ive

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Large-concentration SciPy reference test is NumPy-specific",
)


def _expected_independent_mode_density(kappa):
    circular_mode_density = 1.0 / (2.0 * math.pi * float(ive(0, kappa)))
    return circular_mode_density**2


@pytest.mark.parametrize(
    "distribution_class",
    [ToroidalVonMisesSineDistribution, ToroidalVonMisesCosineDistribution],
    ids=["sine", "cosine"],
)
def test_large_independent_concentrations_have_finite_mode_density(
    distribution_class,
):
    distribution = distribution_class(array([0.3, 1.2]), array([1000.0, 1000.0]), 0.0)
    density = float(np.asarray(to_numpy(distribution.pdf(distribution.mu))))

    assert math.isfinite(density)
    npt.assert_allclose(
        density,
        _expected_independent_mode_density(1000.0),
        rtol=1e-12,
        atol=0.0,
    )


def test_cosine_large_concentration_first_moment_remains_finite():
    mu = array([0.3, 1.2])
    kappa = 1000.0
    distribution = ToroidalVonMisesCosineDistribution(mu, array([kappa, kappa]), 0.0)

    actual = np.asarray(distribution.trigonometric_moment(1))
    expected_magnitude = float(ive(1, kappa) / ive(0, kappa))
    expected = expected_magnitude * np.exp(1j * np.asarray(to_numpy(mu)))

    assert np.all(np.isfinite(actual))
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
