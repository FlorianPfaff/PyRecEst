# pylint: disable=no-name-in-module,no-member
import numpy as np
import pytest
import pyrecest.backend
from pyrecest.backend import array, complex128, conj, float64, sum
from pyrecest.distributions import ComplexWatsonDistribution

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Complex Watson fitting is not supported on JAX",
)


@pytest.fixture(params=["unweighted", "weighted"])
def observations(request):
    rows = np.eye(2, dtype=np.complex128)
    if request.param == "unweighted":
        return np.repeat(rows, [9, 1], axis=0), None
    return rows, array([9.0, 1.0], dtype=float64)


def test_fitting_recovers_dominant_complex_direction(observations):
    rows, weights = observations
    unitary = np.array([[1.0, 1.0], [1j, -1j]]) / np.sqrt(2.0)
    samples = array(rows @ unitary.T, dtype=complex128)
    expected = array(unitary[:, 0], dtype=complex128)
    orthogonal = array(unitary[:, 1], dtype=complex128)

    mu, kappa = ComplexWatsonDistribution.estimate_parameters(samples, weights)
    fitted = ComplexWatsonDistribution.fit(samples, weights)

    assert float(abs(sum(conj(expected) * mu))) == pytest.approx(
        1.0, rel=1e-12, abs=1e-12
    )
    assert float(abs(sum(conj(expected) * fitted.mu))) == pytest.approx(
        1.0, rel=1e-12, abs=1e-12
    )
    assert fitted.kappa == pytest.approx(kappa, rel=1e-12)
    assert kappa > 0.0
    # The D=2 closed-form moment must match the observed 9:1 directional mass.
    assert 1.0 / (-np.expm1(-kappa)) - 1.0 / kappa == pytest.approx(0.9, abs=1e-10)
    assert float(fitted.pdf(expected)) > float(fitted.pdf(orthogonal))


def test_fitted_direction_rotates_with_complex_coordinates(observations):
    rows, weights = observations
    samples = array(rows, dtype=complex128)
    unitary = array([[1.0, 1.0], [1j, -1j]], dtype=complex128) / np.sqrt(2.0)
    mu, kappa = ComplexWatsonDistribution.estimate_parameters(samples, weights)

    rotated_mu, rotated_kappa = ComplexWatsonDistribution.estimate_parameters(
        samples @ unitary.T, weights
    )
    expected = unitary @ mu

    assert float(abs(sum(conj(expected) * rotated_mu))) == pytest.approx(
        1.0, rel=1e-12, abs=1e-12
    )
    assert rotated_kappa == pytest.approx(kappa, rel=1e-12)


def test_fitting_is_invariant_to_individual_sample_phases(observations):
    rows, weights = observations
    unitary = np.array([[1.0, 1.0], [1j, -1j]]) / np.sqrt(2.0)
    rows = rows @ unitary.T
    phases = np.exp(1j * np.linspace(0.2, 2.3, len(rows)))[:, None]
    mu, kappa = ComplexWatsonDistribution.estimate_parameters(
        array(rows, dtype=complex128), weights
    )

    phased_mu, phased_kappa = ComplexWatsonDistribution.estimate_parameters(
        array(rows * phases, dtype=complex128), weights
    )

    assert float(abs(sum(conj(mu) * phased_mu))) == pytest.approx(
        1.0, rel=1e-12, abs=1e-12
    )
    assert phased_kappa == pytest.approx(kappa, rel=1e-12)
