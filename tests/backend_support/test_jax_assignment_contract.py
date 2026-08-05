"""Regression tests for JAX backend assignment helper indexing."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_jax_assignment_accepts_numpy_style_advanced_indices():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

x = backend.zeros((3, 3))
assigned = backend.assignment(x, [5.0, 7.0], [(0, 1), (1, 2)])
summed = backend.assignment_by_sum(backend.ones((3, 3)), [5.0, 7.0], [(0, 1), (1, 2)])

assert backend.to_numpy(assigned).tolist() == [[0.0, 5.0, 0.0], [0.0, 0.0, 7.0], [0.0, 0.0, 0.0]]
assert backend.to_numpy(summed).tolist() == [[1.0, 6.0, 1.0], [1.0, 1.0, 8.0], [1.0, 1.0, 1.0]]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_jax_assignment_preserves_numpy_style_axis_tuple_indices():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

x = backend.zeros((3, 4))
assigned = backend.assignment(x, [5.0, 7.0], ([0, 1], [2, 0]))
summed = backend.assignment_by_sum(backend.ones((3, 4)), [5.0, 7.0], ([0, 1], [2, 0]))

assert backend.to_numpy(assigned).tolist() == [[0.0, 0.0, 5.0, 0.0], [7.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
assert backend.to_numpy(summed).tolist() == [[1.0, 1.0, 6.0, 1.0], [8.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_jax_assignment_accepts_scalar_tuple_index():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

x = backend.zeros((2, 3))
assigned = backend.assignment(x, 9.0, (0, 1))
summed = backend.assignment_by_sum(backend.ones((2, 3)), 4.0, (1, 2))

assert backend.to_numpy(assigned).tolist() == [[0.0, 9.0, 0.0], [0.0, 0.0, 0.0]]
assert backend.to_numpy(summed).tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 5.0]]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_jax_assignment_accepts_list_and_boolean_indices():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

vector = backend.zeros(3)
by_list = backend.assignment(vector, [4.0, 5.0], [0, 2])

matrix = backend.zeros((3, 3))
by_mask = backend.assignment(
    matrix,
    [1.0, 2.0],
    [[True, False, False], [False, True, False], [False, False, False]],
)

assert backend.to_numpy(by_list).tolist() == [4.0, 0.0, 5.0]
assert backend.to_numpy(by_mask).tolist() == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_jax_assignment_uses_full_coordinates_when_count_is_less_than_rank():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

indices = [(0, 1, 2), (1, 2, 3)]
assigned = backend.assignment(
    backend.zeros((2, 3, 4)),
    [5.0, 7.0],
    indices,
)
summed = backend.assignment_by_sum(
    backend.ones((2, 3, 4)),
    [5.0, 7.0],
    indices,
)

expected_assigned = backend.zeros((2, 3, 4))
expected_assigned = expected_assigned.at[0, 1, 2].set(5.0)
expected_assigned = expected_assigned.at[1, 2, 3].set(7.0)
expected_summed = backend.ones((2, 3, 4))
expected_summed = expected_summed.at[0, 1, 2].set(6.0)
expected_summed = expected_summed.at[1, 2, 3].set(8.0)

assert backend.to_numpy(assigned).tolist() == backend.to_numpy(expected_assigned).tolist()
assert backend.to_numpy(summed).tolist() == backend.to_numpy(expected_summed).tolist()
""",
    )

    assert result.returncode == 0, result.stderr
