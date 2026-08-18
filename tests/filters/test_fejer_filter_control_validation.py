import numpy as np
import pytest
import pyrecest.backend
from pyrecest.filters import FejerIdentityFilter


def _skip_unsupported_backend():
    if pyrecest.backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("FejerIdentityFilter is not supported on this backend")


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("adaptive_reduction", "False"),
        ("adaptive_reduction", 1),
        ("min_value_tolerance", "1e-12"),
        ("min_value_tolerance", -1.0e-12),
        ("min_value_tolerance", np.nan),
        ("min_value_tolerance", True),
    ],
)
def test_filter_rejects_silently_coerced_controls(keyword, value):
    _skip_unsupported_backend()

    with pytest.raises(ValueError, match=keyword):
        FejerIdentityFilter((11,), **{keyword: value})


def test_filter_accepts_numpy_scalar_controls():
    _skip_unsupported_backend()

    filt = FejerIdentityFilter(
        (11,),
        adaptive_reduction=np.bool_(False),
        min_value_tolerance=np.float64(0.0),
        oversampling_factor=np.int64(2),
        exponent_search_steps=np.int64(0),
    )

    assert filt.adaptive_reduction is False
    assert filt.min_value_tolerance == 0.0
    assert filt.oversampling_factor == 2
    assert filt.exponent_search_steps == 0
