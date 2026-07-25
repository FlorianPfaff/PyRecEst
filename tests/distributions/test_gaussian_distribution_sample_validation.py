import numpy as np
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


@pytest.mark.parametrize("count", ["3", b"3", np.str_("3"), np.bytes_(b"3")])
def test_gaussian_sample_rejects_text_count(count):
    gaussian = GaussianDistribution(array([0.0]), eye(1))

    with pytest.raises(ValueError, match="n must be a positive integer"):
        gaussian.sample(count)
