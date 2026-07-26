import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.smoothers.hypertoroidal_fourier_smoother import (
    HypertoroidalFourierSmoother,
)
from pyrecest.smoothers.hypertoroidal_grid_smoother import HypertoroidalGridSmoother

N_COEFFICIENTS = 15


def _uniform_identity_hfd(n_coefficients=N_COEFFICIENTS):
    coeffs = np.zeros((n_coefficients,), dtype=complex)
    coeffs[n_coefficients // 2] = 1.0 / (2.0 * np.pi)
    return HypertoroidalFourierDistribution(coeffs, "identity")


def _delta_identity_hfd(n_coefficients=N_COEFFICIENTS):
    coeffs = np.full((n_coefficients,), 1.0 / (2.0 * np.pi), dtype=complex)
    return HypertoroidalFourierDistribution(coeffs, "identity")


def _cosine_identity_hfd(amplitude, phase, n_coefficients=N_COEFFICIENTS):
    coeffs = np.zeros((n_coefficients,), dtype=complex)
    center = n_coefficients // 2
    coeffs[center] = 1.0 / (2.0 * np.pi)
    coeffs[center + 1] = amplitude * np.exp(-1j * phase) / (4.0 * np.pi)
    coeffs[center - 1] = amplitude * np.exp(1j * phase) / (4.0 * np.pi)
    return HypertoroidalFourierDistribution(coeffs, "identity")


def _cosine_sqrt_hfd(amplitude, phase, n_coefficients=N_COEFFICIENTS):
    return HypertoroidalFourierDistribution.from_function(
        lambda x: 1.0 + amplitude * np.cos(x - phase),
        (n_coefficients,),
        "sqrt",
    )


def _sequential_product(distributions, n_coefficients=N_COEFFICIENTS):
    result = _uniform_identity_hfd(n_coefficients)
    for distribution in distributions:
        result = result.multiply(distribution, (n_coefficients,))
    return result


def _filtered_from_likelihoods(likelihoods, n_coefficients=N_COEFFICIENTS):
    filtered = []
    current = _uniform_identity_hfd(n_coefficients)
    for likelihood in likelihoods:
        current = current.multiply(likelihood, (n_coefficients,))
        filtered.append(current)
    return filtered


class TestHypertoroidalFourierSmoother(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_identity_transition_collapses_to_product_of_all_likelihoods(self):
        likelihoods = [
            _cosine_identity_hfd(0.20, 0.1),
            _cosine_identity_hfd(0.25, 0.8),
            _cosine_identity_hfd(0.15, 1.7),
            _cosine_identity_hfd(0.30, 2.4),
        ]
        filtered = _filtered_from_likelihoods(likelihoods)
        expected = _sequential_product(likelihoods)

        smoothed, backward_messages = HypertoroidalFourierSmoother().smooth_identity(
            filtered,
            likelihoods,
            _delta_identity_hfd(),
        )

        self.assertEqual(len(smoothed), len(likelihoods))
        self.assertEqual(len(backward_messages), len(likelihoods))
        for smoothed_state in smoothed:
            npt.assert_allclose(
                smoothed_state.coeff_mat, expected.coeff_mat, rtol=1e-10, atol=1e-12
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_reverse_frequencies_represents_reflected_density(self):
        distribution = _cosine_identity_hfd(0.4, 0.9)
        reflected = HypertoroidalFourierSmoother.reverse_frequencies(distribution)
        xs = np.linspace(0.0, 2.0 * np.pi, 101, endpoint=False)

        npt.assert_allclose(
            reflected.pdf(xs),
            distribution.pdf(np.mod(-xs, 2.0 * np.pi)),
            rtol=1e-11,
            atol=1e-12,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_single_step_sqrt_smoothing_returns_filtered_state(self):
        filtered_state = _cosine_sqrt_hfd(0.25, 0.4)
        likelihood = _cosine_sqrt_hfd(0.10, 1.0)
        smoothed, backward_messages = HypertoroidalFourierSmoother().smooth_identity(
            [filtered_state],
            [likelihood],
            None,
        )
        xs = np.linspace(0.0, 2.0 * np.pi, 101, endpoint=False)

        self.assertEqual(len(smoothed), 1)
        self.assertEqual(len(backward_messages), 1)
        npt.assert_allclose(
            smoothed[0].pdf(xs), filtered_state.pdf(xs), rtol=1e-10, atol=1e-12
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_likelihood_length_is_checked(self):
        likelihoods = [_cosine_identity_hfd(0.2, 0.1), _cosine_identity_hfd(0.3, 0.2)]
        filtered = _filtered_from_likelihoods(likelihoods)
        with self.assertRaises(ValueError):
            HypertoroidalFourierSmoother().smooth_identity(
                filtered, likelihoods[:1], _delta_identity_hfd()
            )


def _grid_likelihood(amplitude, phase, n_grid_points=21):
    return HypertoroidalGridDistribution.from_function(
        lambda xs: 1.0 + amplitude * np.cos(xs[:, 0] - phase),
        n_grid_points,
        grid_type="cartesian_prod",
    )


def _uniform_grid_distribution(n_grid_points=21):
    return HypertoroidalGridDistribution.from_function(
        lambda xs: np.ones(xs.shape[0]),
        n_grid_points,
        grid_type="cartesian_prod",
    )


def _filtered_grid_from_likelihoods(likelihoods):
    filtered = []
    current = _uniform_grid_distribution(likelihoods[0].grid_values.shape[0])
    for likelihood in likelihoods:
        current = current.multiply(likelihood)
        filtered.append(current)
    return filtered


def _identity_grid_transition(reference):
    n_points = reference.grid_values.shape[0]
    cell_volume = (2.0 * np.pi) / n_points
    return TdCondTdGridDistribution(
        reference.get_grid(), np.eye(n_points) / cell_volume
    )


class TestHypertoroidalGridSmoother(unittest.TestCase):
    def test_identity_transition_collapses_to_product_of_all_likelihoods(self):
        likelihoods = [
            _grid_likelihood(0.20, 0.1),
            _grid_likelihood(0.25, 0.8),
            _grid_likelihood(0.15, 1.7),
        ]
        filtered = _filtered_grid_from_likelihoods(likelihoods)
        expected = _uniform_grid_distribution(likelihoods[0].grid_values.shape[0])
        for likelihood in likelihoods:
            expected = expected.multiply(likelihood)

        smoothed, backward_messages = HypertoroidalGridSmoother().smooth(
            filtered,
            likelihoods,
            _identity_grid_transition(filtered[0]),
        )

        self.assertEqual(len(smoothed), len(likelihoods))
        self.assertEqual(len(backward_messages), len(likelihoods))
        for smoothed_state in smoothed:
            npt.assert_allclose(
                smoothed_state.grid_values, expected.grid_values, rtol=1e-12, atol=1e-12
            )

    def test_transition_length_is_checked(self):
        likelihoods = [_grid_likelihood(0.2, 0.1), _grid_likelihood(0.3, 0.2)]
        filtered = _filtered_grid_from_likelihoods(likelihoods)
        transition = _identity_grid_transition(filtered[0])
        with self.assertRaises(ValueError):
            HypertoroidalGridSmoother().smooth(
                filtered, likelihoods, [transition, transition]
            )
