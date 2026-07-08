import numpy as np
import pytest
import pyrecest.backend

from pyrecest.distributions.so3_uniform_distribution import SO3UniformDistribution
from pyrecest.distributions.so3_wigner_distribution import SO3WignerDistribution
from pyrecest.filters.so3_wigner_filter import SO3WignerFilter

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="SO3WignerDistribution currently requires the NumPy backend.",
)


def test_uniform_identity_coefficients_integrate_to_one():
    dist = SO3WignerDistribution.uniform(2, "identity")
    assert np.isclose(dist.integrate(), 1.0)
    assert np.isclose(dist.pdf(np.array([[0.0, 0.0, 0.0, 1.0]]))[0], 1.0 / np.pi**2)


def test_uniform_sqrt_coefficients_integrate_to_one():
    dist = SO3WignerDistribution.uniform(2, "sqrt")
    assert np.isclose(dist.integrate(), 1.0)
    assert np.isclose(dist.pdf(np.array([[0.0, 0.0, 0.0, 1.0]]))[0], 1.0 / np.pi**2)


def test_degree_zero_basis_is_constant():
    value = SO3WignerDistribution.basis_value(0, 0, 0, np.array([0.1]), np.array([0.2]), np.array([0.3]))[0]
    assert np.isclose(value, 1.0 / np.pi)


def test_quadrature_approximates_uniform_distribution():
    dist = SO3WignerDistribution.from_distribution_via_quadrature(SO3UniformDistribution(), degree=1, transformation="identity")
    assert np.isclose(dist.integrate(), 1.0)
    assert np.isclose(dist.pdf(np.array([[0.0, 0.0, 0.0, 1.0]]))[0], 1.0 / np.pi**2, atol=1e-8)


def test_constant_likelihood_update_keeps_uniform_density():
    filt = SO3WignerFilter(2, "identity")

    def likelihood(_measurement, quaternions):
        return np.ones(quaternions.shape[0])

    filt.update_nonlinear(likelihood, None)
    assert np.isclose(filt.filter_state.integrate(), 1.0)
    assert np.isclose(filt.filter_state.pdf(np.array([[0.0, 0.0, 0.0, 1.0]]))[0], 1.0 / np.pi**2, atol=1e-8)


def test_uniform_convolution_keeps_uniform_density():
    filt = SO3WignerFilter(2, "identity")
    filt.predict_identity(SO3WignerDistribution.uniform(2, "identity"))
    assert np.isclose(filt.filter_state.integrate(), 1.0)
    assert np.isclose(filt.filter_state.pdf(np.array([[0.0, 0.0, 0.0, 1.0]]))[0], 1.0 / np.pi**2, atol=1e-8)
