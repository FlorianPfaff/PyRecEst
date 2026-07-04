import numpy as np
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


@pytest.mark.parametrize("count", ["3", np.str_("3")])
def test_mixture_sample_rejects_text_count(count):
    vmf = ToroidalWrappedNormalDistribution(array([1.0, 0.0]), eye(2))
    mixture = HypertoroidalMixture(
        [vmf, vmf.shift(array([1.0, 1.0]))], array([0.5, 0.5])
    )

    with pytest.raises(ValueError, match="n must be a positive integer"):
        mixture.sample(count)
