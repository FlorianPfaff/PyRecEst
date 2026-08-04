"""Regression tests for masked ready-made motion-model controls."""

import numpy as np
import numpy.testing as npt
import pytest

from pyrecest.models import (
    integrated_white_noise_covariance,
    kinematic_transition_matrix,
    singer_transition_matrix,
)


@pytest.mark.parametrize(
    ("call", "field"),
    (
        (
            lambda: kinematic_transition_matrix(
                np.ma.array(1.0, mask=True),
            ),
            "dt",
        ),
        (
            lambda: kinematic_transition_matrix(
                1.0,
                spatial_dim=np.ma.array(2, mask=True),
            ),
            "spatial_dim",
        ),
        (
            lambda: singer_transition_matrix(
                1.0,
                tau=np.ma.masked,
            ),
            "tau",
        ),
    ),
)
def test_motion_models_reject_masked_scalar_controls(call, field):
    with pytest.raises(ValueError, match=field):
        call()


def test_process_noise_rejects_masked_spectral_density_entries():
    invalid_densities = (
        np.ma.array([1.0, 2.0], mask=[False, True]),
        np.array([1.0, np.ma.masked], dtype=object),
    )

    for spectral_density in invalid_densities:
        with pytest.raises(ValueError, match="spectral_density"):
            integrated_white_noise_covariance(
                1.0,
                spatial_dim=2,
                spectral_density=spectral_density,
            )


def test_motion_models_accept_clear_mask_wrappers():
    transition = kinematic_transition_matrix(
        np.ma.array(1.0, mask=False),
        spatial_dim=np.ma.array(1, mask=False),
        derivative_order=np.ma.array(1, mask=False),
    )
    covariance = integrated_white_noise_covariance(
        np.ma.array(1.0, mask=False),
        spatial_dim=2,
        spectral_density=np.ma.array([1.0, 2.0], mask=False),
    )

    npt.assert_allclose(transition, np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert np.all(np.isfinite(np.asarray(covariance)))
