"""Regression tests for PyTorch backend searchsorted support."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_searchsorted_matches_numpy_style_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as raw_pytorch

boundaries = backend.asarray([1.0, 3.0, 5.0])
values = backend.asarray([0.0, 1.0, 2.0, 5.0, 6.0])

left = backend.searchsorted(boundaries, values)
right = backend.searchsorted(boundaries, values, side="right")
raw_result = raw_pytorch.searchsorted([1.0, 3.0, 5.0], [0.0, 1.0, 2.0, 5.0, 6.0])

assert backend.to_numpy(left).tolist() == [0, 0, 1, 2, 3]
assert backend.to_numpy(right).tolist() == [0, 1, 1, 3, 3]
assert raw_pytorch.to_numpy(raw_result).tolist() == [0, 0, 1, 2, 3]

unsorted_boundaries = backend.asarray([5.0, 1.0, 3.0])
sorter = backend.argsort(unsorted_boundaries)
sorted_result = backend.searchsorted(
    unsorted_boundaries,
    backend.asarray([2.0, 4.0]),
    sorter=sorter,
)
assert backend.to_numpy(sorted_result).tolist() == [1, 2]

out = backend.empty((2,), dtype=backend.int32)
returned = backend.searchsorted(
    boundaries,
    backend.asarray([2.0, 4.0]),
    out=out,
    out_int32=True,
)
assert returned is out
assert backend.to_numpy(out).tolist() == [1, 2]

try:
    backend.searchsorted(boundaries, values, side="middle")
except ValueError as exc:
    assert "side" in str(exc)
else:
    raise AssertionError("invalid searchsorted side should fail")
""",
    )

    assert result.returncode == 0, result.stderr
