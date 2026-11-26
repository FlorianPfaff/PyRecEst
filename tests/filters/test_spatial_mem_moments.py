"""Reference integration for the conditional axis update shared by the 3D filters."""

import numpy as np

from pyrecest.backend import array
from pyrecest.filters._spatial_mem_tracker import axis_squared_moments


def test_batched_squared_moments_match_joint_gaussian_quadrature():
    # Three nodes integrate every monomial of degree up to five per variable.
    # A product of four measurements has degree at most four in each of the
    # nine independent coordinates for the axes, source and sensor noise.
    indices = np.indices((3,) * 9).reshape(9, -1).T
    nodes = np.sqrt(3) * (indices - 1)
    weights = np.array([1 / 6, 2 / 3, 1 / 6])[indices].prod(axis=1)
    axis_factor = np.array([[0.4, 0, 0], [-0.15, 0.3, 0], [0.2, 0.1, 0.1]])
    source_factor = np.array([[0.5, 0, 0], [0.2, 0.4, 0], [-0.1, 0.1, 0.3]])
    noise_factor = np.array([[0.3, 0, 0], [-0.1, 0.2, 0], [0.1, 0.05, 0.15]])
    axes = np.array([[1.5, 2.0, 0.7], [0.9, 1.1, 2.3]])
    factors = np.stack([axis_factor, axis_factor / 2])
    means, covariances, cross_covariances = axis_squared_moments(
        array(axes),
        array(factors @ factors.transpose(0, 2, 1)),
        array(source_factor @ source_factor.T),
        array(noise_factor @ noise_factor.T),
    )

    source = nodes[:, 3:6] @ source_factor.T
    noise = nodes[:, 6:] @ noise_factor.T
    for i, factor in enumerate(factors):
        axis_error = nodes[:, :3] @ factor.T
        squares = ((axes[i] + axis_error) * source + noise) ** 2
        mean = weights @ squares
        centered = squares - mean
        np.testing.assert_allclose(means[i], mean, atol=1e-12)
        np.testing.assert_allclose(
            covariances[i], (centered.T * weights) @ centered, atol=1e-12
        )
        np.testing.assert_allclose(
            cross_covariances[i], (axis_error.T * weights) @ centered, atol=1e-12
        )
