import pytest
from pyrecest.backends import SUPPORTED_BACKENDS, get_backend


def test_get_backend_imports_supported_backend():
    backend = get_backend("numpy")
    assert backend.array([1.0]).shape == (1,)
    assert "numpy" in SUPPORTED_BACKENDS


def test_get_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("not-a-backend")
