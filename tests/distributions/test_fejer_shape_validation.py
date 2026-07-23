import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.distributions.hypertorus.fejer import fejer_weights


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (3.5,),
        ("3",),
        (True,),
        (np.bool_(True),),
        "3",
        b"3",
    ],
)
def test_fejer_weights_rejects_coercible_non_integer_shapes(invalid_shape):
    with pytest.raises(ValueError, match="positive odd integers"):
        fejer_weights(invalid_shape)


def test_fejer_weights_accepts_numpy_integer_shape_entries():
    npt.assert_allclose(
        fejer_weights((np.int64(3),)),
        np.array([0.5, 1.0, 0.5]),
    )
