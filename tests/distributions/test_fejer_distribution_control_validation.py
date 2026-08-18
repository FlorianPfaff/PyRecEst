import numpy as np
import pytest
from pyrecest.distributions.hypertorus.fejer_hypertoroidal_fourier_distribution import (
    FejerHypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)


def _uniform_coefficients():
    return np.array([0.0, 1.0 / (2.0 * np.pi), 0.0], dtype=complex)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("adaptive_reduction", "False"),
        ("adaptive_reduction", 1),
        ("min_value_tolerance", "1e-12"),
        ("min_value_tolerance", -1.0e-12),
        ("min_value_tolerance", np.nan),
        ("min_value_tolerance", True),
        ("oversampling_factor", 1.5),
        ("oversampling_factor", True),
        ("exponent_search_steps", 2.5),
        ("exponent_search_steps", -1),
    ],
)
def test_constructor_rejects_silently_coerced_controls(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        FejerHypertoroidalFourierDistribution(
            _uniform_coefficients(), **{keyword: value}
        )


def test_constructor_accepts_numpy_scalar_controls():
    distribution = FejerHypertoroidalFourierDistribution(
        _uniform_coefficients(),
        adaptive_reduction=np.bool_(True),
        min_value_tolerance=np.float64(0.0),
        oversampling_factor=np.int64(2),
        exponent_search_steps=np.int64(0),
    )

    assert distribution.adaptive_reduction is True
    assert distribution.min_value_tolerance == 0.0
    assert distribution.oversampling_factor == 2
    assert distribution.exponent_search_steps == 0


def test_from_fourier_distribution_rejects_truthy_non_boolean_apply_fejer():
    base = HypertoroidalFourierDistribution(
        _uniform_coefficients(), transformation="identity"
    )

    with pytest.raises(ValueError, match="apply_fejer"):
        FejerHypertoroidalFourierDistribution.from_fourier_distribution(
            base, apply_fejer="False"
        )
