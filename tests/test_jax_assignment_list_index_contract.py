import numpy as np
import numpy.testing as npt
import pytest

jnp = pytest.importorskip("jax.numpy")

import pyrecest._backend.jax as jax_backend
from pyrecest.backend_support._jax_assignment_numpy_index_contract import (
    patch_jax_assignment_numpy_index_contract,
)


patch_jax_assignment_numpy_index_contract()


@pytest.mark.parametrize("helper_name", ["assignment", "assignment_by_sum"])
def test_jax_assignment_list_indices_select_first_axis(helper_name):
    helper = getattr(jax_backend, helper_name)

    result = helper(jnp.zeros((2, 3)), 7.0, [0])

    npt.assert_allclose(
        np.asarray(result),
        np.array([[7.0, 7.0, 7.0], [0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("helper_name", ["assignment", "assignment_by_sum"])
def test_jax_assignment_single_coordinate_list_targets_one_entry(helper_name):
    helper = getattr(jax_backend, helper_name)

    result = helper(jnp.zeros((2, 2, 2)), 5.0, [(0, 1, 1)])

    expected = np.zeros((2, 2, 2))
    expected[0, 1, 1] = 5.0
    npt.assert_allclose(np.asarray(result), expected)


def test_jax_assignment_partial_coordinate_list_vectorizes_along_axis():
    result = jax_backend.assignment(
        jnp.zeros((2, 3, 4)),
        2.0,
        [(0, 1)],
        axis=0,
    )

    expected = np.zeros((2, 3, 4))
    expected[:, 0, 1] = 2.0
    npt.assert_allclose(np.asarray(result), expected)
