import warnings

from pyrecest.deprecation import deprecated


def test_deprecated_decorator_emits_standard_warning():
    @deprecated(since="2.3.0", remove_in="3.0.0", replacement="new_function")
    def old_function():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_function() == 1

    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "new_function" in str(caught[0].message)
