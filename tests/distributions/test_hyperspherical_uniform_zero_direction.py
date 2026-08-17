import numpy as np
from pyrecest.distributions import HypersphericalUniformDistribution
from pyrecest.distributions.hypersphere_subset import (
    hyperspherical_uniform_distribution as hyperspherical_uniform_module,
)


def test_hyperspherical_uniform_sample_handles_zero_gaussian_direction(monkeypatch):
    def zero_and_axis_directions(*, size):
        assert size == (2, 2)
        return hyperspherical_uniform_module.array(
            [
                [0.0, 0.0],
                [0.0, 2.0],
            ]
        )

    monkeypatch.setattr(
        hyperspherical_uniform_module.backend_random,
        "normal",
        zero_and_axis_directions,
    )

    distribution = HypersphericalUniformDistribution(dim=1)
    with np.errstate(divide="raise", invalid="raise"):
        samples = np.asarray(distribution.sample(2))

    np.testing.assert_allclose(
        samples,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), np.ones(2))
