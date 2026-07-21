import math

import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions.hypertorus.toroidal_vm_rivest_distribution import (
    ToroidalVMRivestDistribution,
)
from scipy.special import ive

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Large-concentration SciPy reference test is NumPy-specific",
)


def test_large_independent_concentrations_remain_finite():
    mu = array([0.3, 1.2])
    concentration = 1000.0
    dist = ToroidalVMRivestDistribution(
        mu, array([concentration, concentration]), 0.0, 0.0
    )

    density_at_mode = float(dist.pdf(mu))
    expected_density = 1.0 / (4.0 * math.pi**2 * float(ive(0, concentration)) ** 2)
    assert math.isfinite(density_at_mode)
    npt.assert_allclose(density_at_mode, expected_density, rtol=1e-12)

    moment = np.asarray(pyrecest.backend.to_numpy(dist.trigonometric_moment(1)))
    expected_magnitude = float(ive(1, concentration) / ive(0, concentration))
    expected_moment = expected_magnitude * np.exp(
        1j * np.asarray(pyrecest.backend.to_numpy(mu))
    )
    assert np.all(np.isfinite(moment))
    npt.assert_allclose(moment, expected_moment, rtol=1e-12)
