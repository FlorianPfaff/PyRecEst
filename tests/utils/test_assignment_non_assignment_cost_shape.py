import numpy as np
import pytest
import pyrecest.backend
from pyrecest.utils import murty_k_best_assignments


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Murty assignment is not supported on the JAX backend",
)


@pytest.mark.parametrize(
    ("argument_name", "costs"),
    [
        ("row_non_assignment_costs", np.array([[1.0, 2.0]])),
        ("row_non_assignment_costs", np.array([[1.0], [2.0]])),
        ("col_non_assignment_costs", np.array([[1.0, 2.0]])),
        ("col_non_assignment_costs", np.array([[1.0], [2.0]])),
    ],
)
def test_murty_rejects_matrix_shaped_non_assignment_costs(argument_name, costs):
    with pytest.raises(
        ValueError,
        match=rf"{argument_name} must be scalar or one-dimensional",
    ):
        murty_k_best_assignments(
            np.eye(2),
            **{argument_name: costs},
        )


def test_murty_preserves_scalar_and_vector_non_assignment_costs():
    solutions = murty_k_best_assignments(
        np.eye(2),
        row_non_assignment_costs=3.0,
        col_non_assignment_costs=np.array([4.0, 5.0]),
    )

    assert len(solutions) == 1
    np.testing.assert_array_equal(solutions[0]["assignment"], np.array([0, 1]))
    assert solutions[0]["cost"] == pytest.approx(2.0)
