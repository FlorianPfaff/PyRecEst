"""Regression tests for transactional state-space subdivision updates."""

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="These regressions assert NumPy's singular-matrix exception type.",
)


def _make_filter(n_areas=2):
    grid_distribution = HypertoroidalGridDistribution.from_distribution(
        CircularUniformDistribution(), (n_areas,)
    )
    linear_distributions = [
        GaussianDistribution(array([0.0]), eye(1)) for _ in range(n_areas)
    ]
    return StateSpaceSubdivisionFilter(
        StateSpaceSubdivisionGaussianDistribution(
            grid_distribution,
            linear_distributions,
        )
    )


def _snapshot(filter_):
    state = filter_.filter_state
    return (
        np.asarray(state.gd.grid_values).copy(),
        [np.asarray(dist.mu).copy() for dist in state.linear_distributions],
        [np.asarray(dist.C).copy() for dist in state.linear_distributions],
    )


def _assert_snapshot_unchanged(filter_, snapshot):
    grid_values, means, covariances = snapshot
    np.testing.assert_allclose(filter_.filter_state.gd.grid_values, grid_values)
    for distribution, mean, covariance in zip(
        filter_.filter_state.linear_distributions,
        means,
        covariances,
        strict=True,
    ):
        np.testing.assert_allclose(distribution.mu, mean)
        np.testing.assert_allclose(distribution.C, covariance)


def test_single_likelihood_failure_does_not_commit_periodic_weights():
    filter_ = _make_filter()
    snapshot = _snapshot(filter_)
    singular_likelihood = GaussianDistribution(
        array([1.0]),
        array([[0.0]]),
        check_validity=False,
    )

    with pytest.raises(np.linalg.LinAlgError):
        filter_.update(
            likelihood_periodic_grid=array([1.0, 3.0]),
            likelihoods_linear=[singular_likelihood],
        )

    _assert_snapshot_unchanged(filter_, snapshot)


def test_per_cell_failure_does_not_commit_earlier_cell_update():
    filter_ = _make_filter()
    snapshot = _snapshot(filter_)
    valid_likelihood = GaussianDistribution(array([2.0]), eye(1))
    singular_likelihood = GaussianDistribution(
        array([-1.0]),
        array([[0.0]]),
        check_validity=False,
    )

    with pytest.raises(np.linalg.LinAlgError):
        filter_.update(
            likelihood_periodic_grid=array([1.0, 3.0]),
            likelihoods_linear=[valid_likelihood, singular_likelihood],
        )

    _assert_snapshot_unchanged(filter_, snapshot)
