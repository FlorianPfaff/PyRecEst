from __future__ import annotations

import pytest
from pyrecest.utils import pairwise_feature_tensor


@pytest.mark.parametrize(
    ("components", "transforms"),
    [
        ({"feature": [[1.0 + 2.0j]]}, None),
        (
            {"source": [[1.0]]},
            {"feature": lambda _components: [[1.0 + 2.0j]]},
        ),
    ],
    ids=["component", "transform"],
)
def test_pairwise_feature_tensor_rejects_complex_feature_planes(
    components, transforms
) -> None:
    with pytest.raises(
        ValueError,
        match="Feature 'feature' must contain real numeric values",
    ):
        pairwise_feature_tensor(
            components,
            ("feature",),
            transforms=transforms,
        )
