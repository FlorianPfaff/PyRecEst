import pytest
from pyrecest.evaluation.eot_shape_database import StarShapedPolygon


def test_eot_shape_sampling_rejects_negative_counts():
    polygon = StarShapedPolygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])

    with pytest.raises(ValueError, match="nonnegative integer"):
        polygon.sample_on_boundary(-1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        polygon.sample_within(-1)
