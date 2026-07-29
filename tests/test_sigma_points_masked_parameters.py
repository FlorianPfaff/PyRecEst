import numpy as np
import pytest
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


def test_sigma_point_dimensions_reject_masked_scalars():
    masked_dimension = np.ma.array(2, mask=True)

    with pytest.raises(ValueError, match="n must be a scalar"):
        MerweScaledSigmaPoints(
            n=masked_dimension,
            alpha=0.5,
            beta=2.0,
            kappa=0.0,
        )

    with pytest.raises(ValueError, match="n must be a scalar"):
        JulierSigmaPoints(n=masked_dimension, kappa=0.0)


@pytest.mark.parametrize(
    ("parameter_name", "masked_value"),
    (
        ("alpha", np.ma.array(0.5, mask=True)),
        ("beta", np.ma.array(2.0, mask=True)),
        ("kappa", np.ma.array(0.0, mask=True)),
    ),
)
def test_merwe_scaling_parameters_reject_masked_scalars(
    parameter_name,
    masked_value,
):
    parameters = {
        "n": 2,
        "alpha": 0.5,
        "beta": 2.0,
        "kappa": 0.0,
    }
    parameters[parameter_name] = masked_value

    with pytest.raises(ValueError, match=rf"{parameter_name} must be a scalar"):
        MerweScaledSigmaPoints(**parameters)


def test_julier_kappa_rejects_masked_scalars():
    with pytest.raises(ValueError, match="kappa must be a scalar"):
        JulierSigmaPoints(n=2, kappa=np.ma.array(0.0, mask=True))


def test_sigma_point_parameters_accept_unmasked_masked_array_scalars():
    merwe = MerweScaledSigmaPoints(
        n=np.ma.array(2, mask=False),
        alpha=np.ma.array(0.5, mask=False),
        beta=np.ma.array(2.0, mask=False),
        kappa=np.ma.array(0.0, mask=False),
    )
    julier = JulierSigmaPoints(
        n=np.ma.array(2, mask=False),
        kappa=np.ma.array(0.0, mask=False),
    )

    assert merwe.n == 2
    assert merwe.alpha == 0.5
    assert merwe.beta == 2.0
    assert merwe.kappa == 0.0
    assert julier.n == 2
    assert julier.kappa == 0.0
