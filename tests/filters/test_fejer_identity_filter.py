import importlib

import numpy.testing as npt
import pytest

PACKAGE = "py" + "recest"


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip("optional recursive-estimation dependency is not installed")


def test_filter_initializes_uniform_identity_state():
    backend = _optional_module(PACKAGE + ".backend")
    if backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("not supported on this backend")

    from pyrecest.distributions import FejerHypertoroidalFourierDistribution
    from pyrecest.filters import FejerIdentityFilter

    filt = FejerIdentityFilter((11,))
    assert isinstance(filt.filter_state, FejerHypertoroidalFourierDistribution)
    assert filt.filter_state.transformation == "identity"

    xs = backend.linspace(0.0, 2.0 * backend.pi, 256, endpoint=False)
    integral = float(filt.filter_state.pdf(xs).sum()) * float(2.0 * backend.pi / 256)
    npt.assert_allclose(integral, 1.0, atol=1e-4)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("oversampling_factor", 1.5),
        ("exponent_search_steps", 2.5),
    ],
)
def test_filter_rejects_fractional_integer_controls(keyword, value):
    backend = _optional_module(PACKAGE + ".backend")
    if backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("not supported on this backend")

    from pyrecest.filters import FejerIdentityFilter

    with pytest.raises(ValueError, match=keyword):
        FejerIdentityFilter((11,), **{keyword: value})


def test_predict_and_update_identity_1d_are_normalized():
    backend = _optional_module(PACKAGE + ".backend")
    if backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("not supported on this backend")
    distributions = _optional_module(PACKAGE + ".distributions")

    from pyrecest.filters import FejerIdentityFilter

    wrapped_normal = distributions.WrappedNormalDistribution
    filt = FejerIdentityFilter((15,))
    filt.filter_state = wrapped_normal(backend.array(1.0), backend.array(0.4))
    filt.predict_identity(wrapped_normal(backend.array(0.0), backend.array(0.2)))
    filt.update_identity(wrapped_normal(backend.array(0.0), backend.array(0.7)), backend.array([1.3]))

    xs = backend.linspace(0.0, 2.0 * backend.pi, 256, endpoint=False)
    integral = float(filt.filter_state.pdf(xs).sum()) * float(2.0 * backend.pi / 256)
    npt.assert_allclose(integral, 1.0, atol=1e-4)
