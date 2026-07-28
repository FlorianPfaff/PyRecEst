import numpy as np
import numpy.testing as npt
import pytest
from pyrecest import backend
from pyrecest.distributions import AbstractHypertoroidalDistribution

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="SciPy nquad integration is supported only on the NumPy backend",
)


def test_integrate_fun_over_domain_part_accepts_python_scalar():
    result = AbstractHypertoroidalDistribution.integrate_fun_over_domain_part(
        lambda x: x,
        np.array([[0.0, 1.0]]),
    )

    npt.assert_allclose(result, 0.5)
