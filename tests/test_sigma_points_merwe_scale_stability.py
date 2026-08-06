import numpy as np
import pytest
from pyrecest.backend import asarray, to_numpy
from pyrecest.sampling import MerweScaledSigmaPoints


def test_merwe_scale_avoids_lambda_cancellation():
    alpha = 2.0e-8
    points = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=2.0, kappa=0.0)

    sigma_points = to_numpy(
        points.sigma_points(asarray(np.zeros(1)), asarray(np.eye(1)))
    )
    mean_weights = to_numpy(points.Wm)
    scale = alpha * alpha
    expected_weights = np.array(
        [
            (scale - 1.0) / scale,
            0.5 / scale,
            0.5 / scale,
        ]
    )

    np.testing.assert_allclose(
        sigma_points[:, 0],
        np.array([0.0, alpha, -alpha]),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        mean_weights,
        expected_weights,
        rtol=1.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize("alpha", [1.0e-200, 1.0e200], ids=["underflow", "overflow"])
def test_merwe_rejects_nonrepresentable_sigma_point_scales(alpha):
    with pytest.raises(ValueError, match="finite, positive sigma-point scale"):
        MerweScaledSigmaPoints(n=1, alpha=alpha, beta=2.0, kappa=0.0)


def test_merwe_rejects_mean_weights_that_cannot_represent_unit_sum():
    with pytest.raises(ValueError, match="finite, normalized mean weights"):
        MerweScaledSigmaPoints(n=1, alpha=1.0e-8, beta=2.0, kappa=0.0)
