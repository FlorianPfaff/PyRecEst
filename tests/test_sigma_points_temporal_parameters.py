import numpy as np
import pytest
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints

_TEMPORAL_VALUES = (
    np.timedelta64(2, "ns"),
    np.timedelta64(2, "us"),
    np.datetime64("1970-01-01T00:00:00.000000002"),
    np.array(np.timedelta64(2, "ns"), dtype=object),
)


@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_sigma_point_dimensions_reject_temporal_scalars(temporal_value):
    with pytest.raises(ValueError, match="n must be a scalar"):
        MerweScaledSigmaPoints(
            n=temporal_value,
            alpha=0.5,
            beta=2.0,
            kappa=0.0,
        )

    with pytest.raises(ValueError, match="n must be a scalar"):
        JulierSigmaPoints(n=temporal_value, kappa=0.0)


@pytest.mark.parametrize("parameter_name", ("alpha", "beta", "kappa"))
@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_merwe_scaling_parameters_reject_temporal_scalars(
    parameter_name,
    temporal_value,
):
    parameters = {
        "n": 2,
        "alpha": 0.5,
        "beta": 2.0,
        "kappa": 0.0,
    }
    parameters[parameter_name] = temporal_value

    with pytest.raises(ValueError, match=rf"{parameter_name} must be a scalar"):
        MerweScaledSigmaPoints(**parameters)


@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_julier_kappa_rejects_temporal_scalars(temporal_value):
    with pytest.raises(ValueError, match="kappa must be a scalar"):
        JulierSigmaPoints(n=2, kappa=temporal_value)
