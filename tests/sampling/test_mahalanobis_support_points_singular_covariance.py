from __future__ import annotations

import numpy as np
from pyrecest.sampling import mahalanobis_support_points


def test_singular_covariance_does_not_project_rays_onto_its_support() -> None:
    mean = np.asarray([1.0, -2.0])
    covariance = np.diag([4.0, 0.0])
    directions = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    support = mahalanobis_support_points(mean, covariance, directions)

    assert np.allclose(support[0], [3.0, -2.0])
    assert np.allclose(support[1:], mean)
