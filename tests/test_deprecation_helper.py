import asyncio
import inspect
import warnings

import pytest
from pyrecest.deprecation import deprecated


def test_deprecated_decorator_emits_standard_warning():
    @deprecated(since="2.3.0", remove_in="3.0.0", replacement="new_function")
    def legacy_function():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert legacy_function() == 1

    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "new_function" in str(caught[0].message)


def test_deprecated_decorator_preserves_async_function_contract():
    @deprecated(since="2.3.0", remove_in="3.0.0", replacement="new_async_function")
    async def legacy_async_function(value):
        return value + 1

    assert inspect.iscoroutinefunction(legacy_async_function)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert asyncio.run(legacy_async_function(1)) == 2

    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "new_async_function" in str(caught[0].message)


def test_deprecated_decorator_rejects_blank_since():
    with pytest.raises(ValueError, match="since must be a non-empty string"):
        deprecated(since=" ", remove_in="3.0.0")


def test_deprecated_decorator_rejects_blank_remove_in():
    with pytest.raises(ValueError, match="remove_in must be a non-empty string"):
        deprecated(since="2.3.0", remove_in=" ")


def test_deprecated_decorator_rejects_blank_replacement():
    with pytest.raises(ValueError, match="replacement must be a non-empty string"):
        deprecated(since="2.3.0", remove_in="3.0.0", replacement=" ")


def test_deprecated_decorator_strips_metadata_whitespace():
    @deprecated(since=" 2.3.0 ", remove_in=" 3.0.0 ", replacement=" new_function ")
    def legacy_function():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert legacy_function() == 1

    message = str(caught[0].message)
    assert "PyRecEst 2.3.0" in message
    assert "PyRecEst 3.0.0" in message
    assert "new_function" in message
