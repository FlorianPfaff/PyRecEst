import numpy as np
import pyrecest.backend as backend
import pytest
from pyrecest.utils.assignment import murty_k_best_assignments


def _assignment_tuple(solution):
    return tuple(int(index) for index in solution["assignment"])


def test_murty_forced_prefix_subproblems_do_not_repeat_assignments():
    if backend.__backend_name__ == "jax":
        pytest.skip("Murty k-best assignment is not supported on the JAX backend")

    cost_matrix = np.asarray(
        [
            [0.0, 100.0],
            [1.0, 2.0],
        ]
    )

    solutions = murty_k_best_assignments(cost_matrix, k=5)
    assignments = [_assignment_tuple(solution) for solution in solutions]

    assert len(assignments) == 5
    assert len(assignments) == len(set(assignments))
    assert set(assignments) == {
        (0, -1),
        (-1, -1),
        (-1, 0),
        (0, 1),
        (-1, 1),
    }
    assert [solution["cost"] for solution in solutions] == sorted(
        solution["cost"] for solution in solutions
    )
