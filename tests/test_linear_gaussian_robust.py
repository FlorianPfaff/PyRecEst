import pytest
from pyrecest.backend import allclose, array
from pyrecest.filters import student_t_covariance_scale


def test_student_t_covariance_scale_matches_formula():
    scale = student_t_covariance_scale(
        normalized_innovation_squared=14.0,
        measurement_dim=2,
        dof=4.0,
    )

    assert allclose(scale, (4.0 + 14.0) / (4.0 + 2.0))


def test_student_t_covariance_scale_clamps_inliers_to_one_by_default():
    scale = student_t_covariance_scale(
        normalized_innovation_squared=0.5,
        measurement_dim=2,
        dof=4.0,
    )

    assert allclose(scale, 1.0)


def test_student_t_covariance_scale_can_return_subunit_inlier_scale():
    scale = student_t_covariance_scale(
        normalized_innovation_squared=0.5,
        measurement_dim=2,
        dof=4.0,
        min_scale=0.0,
    )

    assert allclose(scale, (4.0 + 0.5) / (4.0 + 2.0))


def test_student_t_covariance_scale_vectorizes_over_nis():
    nis = array([0.0, 6.0, 20.0])
    scale = student_t_covariance_scale(nis, measurement_dim=2, dof=4.0)
    expected = array(
        [
            1.0,
            (4.0 + 6.0) / (4.0 + 2.0),
            (4.0 + 20.0) / (4.0 + 2.0),
        ]
    )

    assert allclose(scale, expected)


def test_student_t_covariance_scale_validates_arguments():
    with pytest.raises(ValueError, match="measurement_dim"):
        student_t_covariance_scale(
            normalized_innovation_squared=1.0,
            measurement_dim=0,
        )
    with pytest.raises(ValueError, match="dof"):
        student_t_covariance_scale(
            normalized_innovation_squared=1.0,
            measurement_dim=2,
            dof=2.0,
        )
    with pytest.raises(ValueError, match="min_scale"):
        student_t_covariance_scale(
            normalized_innovation_squared=1.0,
            measurement_dim=2,
            min_scale=-1.0,
        )
