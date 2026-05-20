from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    get_api_backend_support,
    iter_api_backend_capabilities,
)


def test_public_api_capability_rows_use_known_support_levels():
    assert API_BACKEND_CAPABILITIES
    for api_name, row in iter_api_backend_capabilities():
        assert api_name
        for backend_name in ("numpy", "pytorch", "jax"):
            assert row[backend_name] in BACKEND_SUPPORT_LEVELS
        assert row.get("notes")


def test_get_api_backend_support_returns_copy():
    row = get_api_backend_support("KalmanFilter")
    row["numpy"] = "mutated"
    assert API_BACKEND_CAPABILITIES["KalmanFilter"]["numpy"] == "supported"
