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
def test_sigma_points_reject_complex_state_mean(points):
    with pytest.raises(ValueError, match="x must contain real values"):
        points.sigma_points(
            np.array([0.0 + 1.0j, 1.0]),
            np.eye(2),
        )


@pytest.mark.parametrize(
    "points",
    [
        JulierSigmaPoints(n=2, kappa=1.0),
        MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
    ],
    ids=["julier", "merwe"],
)
def test_sigma_points_reject_complex_covariance(points):
    with pytest.raises(ValueError, match="P must contain real values"):
        points.sigma_points(
            np.array([0.0, 1.0]),
            np.eye(2, dtype=complex),
        )
