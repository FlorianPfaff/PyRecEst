"""Regression tests for raw JAX assignment helper empty-index handling."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_raw_jax_assignment_helpers_treat_empty_indices_as_noop():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import importlib

import numpy as np
from pyrecest.backend import to_numpy

raw_jax = importlib.import_module("pyrecest._backend.jax")
original = raw_jax.array([1.0, 2.0, 3.0])

assigned = raw_jax.assignment(original, 99.0, [])
added = raw_jax.assignment_by_sum(original, 99.0, [])
array_like = raw_jax.assignment([1.0, 2.0, 3.0], 99.0, [])

np.testing.assert_allclose(to_numpy(assigned), [1.0, 2.0, 3.0])
np.testing.assert_allclose(to_numpy(added), [1.0, 2.0, 3.0])
np.testing.assert_allclose(to_numpy(array_like), [1.0, 2.0, 3.0])
""",
    )

    assert result.returncode == 0, result.stderr
