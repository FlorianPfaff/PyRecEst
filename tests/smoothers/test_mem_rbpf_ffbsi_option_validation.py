"""Regression tests for strict MEM-RBPF FFBSi option validation."""

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest import backend
from pyrecest.smoothers import MEMRBPFFFBSiSmoother, MEMRBPFForwardRecord

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="MEM-RBPF FFBSi tests use NumPy sampling paths",
)


def _single_particle_record() -> MEMRBPFForwardRecord:
    return MEMRBPFForwardRecord(
        kinematic_state=np.array([0.0]),
        covariance=np.array([[1.0]]),
        theta=np.array([0.0]),
        axis_mean=np.array([[1.0, 0.5]]),
        axis_covariance=np.array([0.1 * np.eye(2)]),
        weights=np.array([1.0]),
    )


@pytest.mark.parametrize(
    ("option_name", "invalid_value", "message"),
    (
        ("n_trajectories", 1.5, "n_trajectories must be a positive integer"),
        ("n_trajectories", True, "n_trajectories must be a positive integer"),
        (
            "n_trajectories",
            np.timedelta64(3, "ns"),
            "n_trajectories must be a positive integer",
        ),
        (
            "angle_wrap_terms",
            -1,
            "angle_wrap_terms must be a non-negative integer",
        ),
        (
            "angle_wrap_terms",
            0.5,
            "angle_wrap_terms must be a non-negative integer",
        ),
        (
            "angle_wrap_terms",
            np.ma.array(2, mask=True),
            "angle_wrap_terms must be a non-negative integer",
        ),
        (
            "axis_floor",
            np.nan,
            "axis_floor must be a positive finite real scalar",
        ),
        (
            "axis_floor",
            np.inf,
            "axis_floor must be a positive finite real scalar",
        ),
        (
            "axis_floor",
            True,
            "axis_floor must be a positive finite real scalar",
        ),
        (
            "axis_floor",
            np.ma.array(1e-9, mask=True),
            "axis_floor must be a positive finite real scalar",
        ),
    ),
)
def test_constructor_rejects_lossy_or_invalid_options(
    option_name, invalid_value, message
):
    with pytest.raises(ValueError, match=message):
        MEMRBPFFFBSiSmoother(**{option_name: invalid_value})


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"n_trajectories": 1.5}, "n_trajectories must be a positive integer"),
        ({"angle_wrap_terms": -1}, "angle_wrap_terms must be a non-negative integer"),
        ({"full_axis_lengths": "False"}, "full_axis_lengths must be a bool"),
    ),
)
def test_smooth_rejects_invalid_per_call_options(options, message):
    smoother = MEMRBPFFFBSiSmoother(n_trajectories=1, sample_axis=False)

    with pytest.raises(ValueError, match=message):
        smoother.smooth([_single_particle_record()], rng=0, **options)


def test_exact_numpy_scalar_options_remain_supported():
    smoother = MEMRBPFFFBSiSmoother(
        n_trajectories=np.int64(1),
        sample_axis=False,
        angle_wrap_terms=np.array(0),
        axis_floor=np.ma.array(1e-8, mask=False),
    )

    result = smoother.smooth(
        [_single_particle_record()],
        rng=0,
        n_trajectories=np.int64(2),
        angle_wrap_terms=np.array(0),
        full_axis_lengths=np.bool_(False),
    )

    assert smoother.axis_floor == 1e-8
    assert result.sample_states.shape == (2, 1, 4)
    npt.assert_allclose(result.sample_states[:, 0, -2:], [[1.0, 0.5], [1.0, 0.5]])
