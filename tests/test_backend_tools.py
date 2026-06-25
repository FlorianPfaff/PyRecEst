import os

import pyrecest
import pytest


def test_backend_tools_report_active_backend():
    active = pyrecest.get_backend_name()

    assert pyrecest.is_backend(active)
    pyrecest.assert_backend(active)


def test_assert_backend_rejects_unexpected_backend():
    active = pyrecest.get_backend_name()
    unexpected = "jax" if active != "jax" else "numpy"

    with pytest.raises(RuntimeError):
        pyrecest.assert_backend(unexpected)


@pytest.mark.parametrize(
    "expected", [(), ("",), " ", (" numpy",), ("numpy ",), ("numpy", 1), 1]
)
def test_assert_backend_rejects_invalid_expected_names(expected):
    with pytest.raises(ValueError, match="expected"):
        pyrecest.assert_backend(expected)


def test_warn_if_backend_env_changed(monkeypatch):
    active = pyrecest.get_backend_name()
    changed = "jax" if active != "jax" else "numpy"
    monkeypatch.setenv("PYRECEST_BACKEND", changed)

    with pytest.warns(RuntimeWarning):
        pyrecest.warn_if_backend_env_changed()

    monkeypatch.setenv("PYRECEST_BACKEND", active)
    pyrecest.warn_if_backend_env_changed()
