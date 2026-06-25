import numpy as np
import pytest
from pyrecest.evaluation import (
    classify_inside_outside,
    surface_band_mask,
    surface_band_probability_from_signed_distance,
    surface_gradients,
    surface_residuals,
    surface_variances,
)
from pyrecest.protocols import (
    SupportsProbabilisticScalarField,
    SupportsScalarField,
    SupportsScalarFieldGradient,
)


class PlaneField:
    @property
    def input_dim(self) -> int:
        return 3

    def value(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points[..., 2] - 1.0

    def gradient(self, points):
        points = np.asarray(points, dtype=np.float64)
        gradient = np.zeros(points.shape, dtype=np.float64)
        gradient[..., 2] = 1.0
        return gradient

    def variance_at(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.full(points.shape[:-1], 0.04, dtype=np.float64)


class NonCallableField:
    value = 1.0
    gradient = 1.0
    variance_at = 1.0


def test_scalar_field_protocols_are_structural() -> None:
    field = PlaneField()

    assert isinstance(field, SupportsScalarField)
    assert isinstance(field, SupportsScalarFieldGradient)
    assert isinstance(field, SupportsProbabilisticScalarField)


def test_surface_residual_gradient_and_variance_helpers() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.5]], dtype=np.float64)
    field = PlaneField()

    assert np.allclose(surface_residuals(field, points), [0.0, 0.5])
    assert np.allclose(
        surface_gradients(field, points),
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(surface_variances(field, points), [0.04, 0.04])


@pytest.mark.parametrize(
    ("helper", "method_name"),
    [
        (surface_residuals, "value"),
        (surface_gradients, "gradient"),
        (surface_variances, "variance_at"),
    ],
)
def test_surface_helpers_reject_non_callable_protocol_members(
    helper, method_name
) -> None:
    points = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)

    with pytest.raises(TypeError, match=f"surface must implement {method_name}"):
        helper(NonCallableField(), points)


def test_surface_band_mask_and_inside_outside_classification() -> None:
    values = np.asarray([-0.2, -0.01, 0.0, 0.04, 0.3, np.nan])

    assert surface_band_mask(values, 0.05).tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert classify_inside_outside(values).tolist() == [-1, -1, 0, 1, 1, 0]
    assert classify_inside_outside(values, negative_inside=False).tolist() == [
        1,
        1,
        0,
        -1,
        -1,
        0,
    ]


def test_classify_inside_outside_accepts_numpy_boolean_scalar() -> None:
    values = np.asarray([-0.2, 0.0, 0.3])

    assert classify_inside_outside(
        values, negative_inside=np.bool_(False)
    ).tolist() == [
        1,
        0,
        -1,
    ]


@pytest.mark.parametrize("bad_flag", ["False", "true", 0, 1, None, [False]])
def test_classify_inside_outside_rejects_non_boolean_sign_flag(bad_flag) -> None:
    with pytest.raises(TypeError, match="negative_inside must be a boolean"):
        classify_inside_outside(np.asarray([1.0]), negative_inside=bad_flag)


def test_surface_band_probability_from_signed_distance() -> None:
    probability = surface_band_probability_from_signed_distance(
        np.asarray([0.0, 0.2], dtype=np.float64),
        np.asarray([0.05, 0.05], dtype=np.float64),
        0.1,
    )

    assert probability[0] > 0.9
    assert 0.0 <= probability[1] < probability[0]


def test_surface_band_probability_accepts_zero_std_floor() -> None:
    probability = surface_band_probability_from_signed_distance(
        np.asarray([0.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 0.5], dtype=np.float64),
        1.0,
        min_std=0.1,
    )

    assert probability.shape == (2,)
    assert np.all(np.isfinite(probability))
    assert np.all((0.0 <= probability) & (probability <= 1.0))
    assert probability[0] > probability[1]


def test_surface_band_probability_rejects_invalid_scales() -> None:
    with pytest.raises(ValueError, match="threshold"):
        surface_band_mask(np.asarray([0.0]), 0.0)
    with pytest.raises(ValueError, match="epsilon"):
        surface_band_probability_from_signed_distance(
            np.asarray([0.0]),
            np.asarray([1.0]),
            0.0,
        )
    with pytest.raises(ValueError, match="min_std"):
        surface_band_probability_from_signed_distance(
            np.asarray([0.0]),
            np.asarray([1.0]),
            0.1,
            min_std=0.0,
        )
    with pytest.raises(ValueError, match="distance_std must be non-negative"):
        surface_band_probability_from_signed_distance(
            np.asarray([0.0]),
            np.asarray([-1.0]),
            0.1,
        )
    for invalid_std in (np.nan, np.inf, -np.inf):
        with pytest.raises(
            ValueError,
            match="distance_std must contain only finite values",
        ):
            surface_band_probability_from_signed_distance(
                np.asarray([0.0]),
                np.asarray([invalid_std]),
                0.1,
            )
