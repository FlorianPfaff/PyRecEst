import numpy as np
import pytest
import pyrecest.backend

from pyrecest.backend import array
from pyrecest.utils.point_set_registration import (
    estimate_transform,
    joint_registration_assignment,
)

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Point-set registration is not supported on JAX.",
)

_SOURCE = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_REFLECTED = array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("invalid_flag", ["False", "True", 0, 1, None])
def test_estimate_transform_rejects_non_boolean_allow_reflection(invalid_flag):
    with pytest.raises(TypeError, match="allow_reflection must be a boolean"):
        estimate_transform(
            _SOURCE,
            _REFLECTED,
            model="rigid",
            allow_reflection=invalid_flag,
        )


def test_joint_registration_rejects_truthy_string_allow_reflection():
    with pytest.raises(TypeError, match="allow_reflection must be a boolean"):
        joint_registration_assignment(
            _SOURCE,
            _REFLECTED,
            model="rigid",
            allow_reflection="False",
        )


def test_numpy_boolean_allow_reflection_is_accepted():
    transform = estimate_transform(
        _SOURCE,
        _REFLECTED,
        model="rigid",
        allow_reflection=np.bool_(True),
    )

    assert float(np.linalg.det(np.asarray(transform.matrix))) < 0.0
