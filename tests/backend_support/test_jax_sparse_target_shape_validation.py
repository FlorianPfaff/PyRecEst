"""Validation tests for JAX sparse reconstruction target shapes."""

from __future__ import annotations

import numpy as np
import pytest
from pyrecest.backend_support._jax_array_from_sparse_contract import (
    _normalize_sparse_target_shape,
)


def test_sparse_target_shape_accepts_exact_integer_dimensions():
    assert _normalize_sparse_target_shape(4) == (4,)
    assert _normalize_sparse_target_shape(np.array(4, dtype=np.int64)) == (4,)
    assert _normalize_sparse_target_shape((2, np.int32(3))) == (2, 3)
    assert _normalize_sparse_target_shape(()) == ()


@pytest.mark.parametrize(
    "target_shape",
    [
        2.5,
        (2.5,),
        (np.array(2.0),),
        True,
        (np.bool_(False),),
        (np.inf,),
        "12",
        ("2",),
    ],
)
def test_sparse_target_shape_rejects_noninteger_dimensions(target_shape):
    with pytest.raises(TypeError, match="integer"):
        _normalize_sparse_target_shape(target_shape)


def test_sparse_target_shape_preserves_negative_dimension_error():
    with pytest.raises(ValueError, match="negative dimensions"):
        _normalize_sparse_target_shape((2, -1))
