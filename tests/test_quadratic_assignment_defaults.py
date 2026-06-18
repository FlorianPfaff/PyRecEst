import importlib.util

import numpy as np
from pyrecest._backend import numpy as numpy_backend

pytorch_backend = None
if importlib.util.find_spec("torch") is not None:
    from pyrecest._backend import pytorch as pytorch_backend


def _problem_matrices():
    adjacency = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ]
    )
    permuted = adjacency[[1, 2, 0]][:, [1, 2, 0]]
    return adjacency, permuted


def test_numpy_quadratic_assignment_accepts_default_options():
    adjacency, permuted = _problem_matrices()

    assignment = numpy_backend.linalg.quadratic_assignment(adjacency, permuted)

    assert sorted(assignment) == [0, 1, 2]


def test_numpy_quadratic_assignment_still_accepts_options_dict():
    adjacency, permuted = _problem_matrices()

    assignment = numpy_backend.linalg.quadratic_assignment(
        adjacency,
        permuted,
        options={"maximize": False},
    )

    assert sorted(assignment) == [0, 1, 2]


def test_pytorch_quadratic_assignment_accepts_default_options():
    if pytorch_backend is None:
        return
    adjacency, permuted = _problem_matrices()

    assignment = pytorch_backend.linalg.quadratic_assignment(
        pytorch_backend.array(adjacency),
        pytorch_backend.array(permuted),
    )

    assert sorted(assignment) == [0, 1, 2]
