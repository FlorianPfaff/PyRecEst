from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import diagnostic_from_record, innovation_diagnostic


@pytest.mark.parametrize(
    ("field", "residual", "innovation_covariance"),
    (
        (
            "residual",
            np.ma.array([1.0, 999.0], mask=[False, True]),
            np.eye(2),
        ),
        (
            "residual",
            [1.0, np.ma.masked],
            np.eye(2),
        ),
        (
            "innovation_covariance",
            np.array([1.0, 2.0]),
            np.ma.array(
                [[1.0, 0.0], [0.0, 999.0]],
                mask=[[False, False], [False, True]],
            ),
        ),
    ),
)
def test_innovation_diagnostic_rejects_masked_arrays(
    field, residual, innovation_covariance
) -> None:
    with pytest.raises(ValueError, match=field):
        innovation_diagnostic(residual, innovation_covariance)


def test_diagnostic_from_record_rejects_masked_serialized_arrays() -> None:
    with pytest.raises(ValueError, match="residual"):
        diagnostic_from_record(
            {
                "measurement_dim": 2,
                "residual": np.ma.array(
                    [1.0, 999.0],
                    mask=[False, True],
                ),
                "innovation_covariance": np.eye(2),
            }
        )


def test_innovation_diagnostic_accepts_clear_mask_wrappers() -> None:
    diagnostic = innovation_diagnostic(
        np.ma.array([1.0, 2.0], mask=False),
        np.ma.array(np.eye(2), mask=False),
    )

    assert diagnostic.measurement_dim == 2
    assert diagnostic.nis == pytest.approx(5.0)
    np.testing.assert_array_equal(diagnostic.residual, np.array([1.0, 2.0]))
