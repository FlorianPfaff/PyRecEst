import pytest
from pyrecest.stability import (
    PublicAPIStatus,
    get_public_api_status,
    iter_public_api_status,
    stability,
)


def test_registered_public_api_status():
    status = get_public_api_status("KalmanFilter")
    assert status is not None
    assert status.level == "stable"
    assert list(iter_public_api_status())


def test_public_api_status_rejects_unknown_stability_level():
    with pytest.raises(ValueError, match="Unknown stability level"):
        PublicAPIStatus("ExampleAPI", "public")  # type: ignore[arg-type]


def test_stability_decorator_attaches_metadata():
    @stability("experimental", since="2.3.0", notes="test helper")
    def sample():
        return 1

    assert sample() == 1
    assert sample.__pyrecest_stability__.level == "experimental"


def test_stability_decorator_attaches_public_api_status():
    @stability("experimental", since="2.4.0", notes="example")
    def example_function():
        return None

    status = example_function.__pyrecest_stability__

    assert isinstance(status, PublicAPIStatus)
    assert status.name.endswith("example_function")
    assert status.level == "experimental"
    assert status.since == "2.4.0"
    assert status.notes == "example"


def test_registered_public_api_status_rows_have_valid_levels():
    rows = list(iter_public_api_status())

    assert rows
    assert get_public_api_status("KalmanFilter") in rows
    assert all(isinstance(row, PublicAPIStatus) for row in rows)
