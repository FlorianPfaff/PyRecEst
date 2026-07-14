import numpy as np
import pytest

from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


@pytest.mark.parametrize(
    "points",
    [
        JulierSigmaPoints(n=2, kappa=1.0),
        MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
    ],
    ids=["julier", "merwe"],
)
def test_sigma_points_reject_object_array_with_complex_state(points):
    mean = np.array([0.0 + 1.0j, 1.0], dtype=object)

    with pytest.raises(ValueError, match="x must contain real values"):
        points.sigma_points(mean, np.eye(2))


@pytest.mark.parametrize(
    "points",
    [
        JulierSigmaPoints(n=2, kappa=1.0),
        MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
    ],
    ids=["julier", "merwe"],
)
def test_sigma_points_reject_object_array_with_complex_covariance(points):
    covariance = np.array(
        [[1.0 + 0.0j, 0.0], [0.0, 1.0]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="P must contain real values"):
        points.sigma_points(np.zeros(2), covariance)
