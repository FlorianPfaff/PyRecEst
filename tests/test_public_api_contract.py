import importlib

import pytest

PUBLIC_MODULES_WITH_ALL = (
    "pyrecest",
    "pyrecest.filters",
)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES_WITH_ALL)
def test_public_all_has_no_duplicates(module_name):
    module = importlib.import_module(module_name)
    exports = tuple(getattr(module, "__all__", ()))

    duplicates = sorted({name for name in exports if exports.count(name) > 1})
    assert duplicates == []


@pytest.mark.parametrize("module_name", PUBLIC_MODULES_WITH_ALL)
def test_public_all_exports_resolve(module_name):
    module = importlib.import_module(module_name)
    exports = tuple(getattr(module, "__all__", ()))

    missing = sorted(name for name in exports if not hasattr(module, name))
    assert missing == []


def test_default_backend_public_smoke_imports():
    from pyrecest.backend import array, diag
    from pyrecest.filters import KalmanFilter

    kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))
    estimate = kalman_filter.get_point_estimate()

    assert estimate is not None
