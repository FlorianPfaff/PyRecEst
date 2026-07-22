import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


def test_gaussian_mixture_pdf_preserves_nested_batch_shape():
    first = GaussianDistribution(array([0.0, 0.0]), diag(array([1.0, 2.0])))
    second = GaussianDistribution(array([1.0, -1.0]), diag(array([2.0, 1.0])))
    mixture = GaussianMixture([first, second], array([0.4, 0.6]))
    points = array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.5], [0.5, -0.5], [2.0, -1.0]],
        ]
    )

    actual = mixture.pdf(points)
    expected = 0.4 * first.pdf(points) + 0.6 * second.pdf(points)

    assert actual.shape == (2, 3)
    npt.assert_allclose(
        np.asarray(pyrecest.backend.to_numpy(actual)),
        np.asarray(pyrecest.backend.to_numpy(expected)),
        rtol=1e-12,
        atol=0.0,
    )
