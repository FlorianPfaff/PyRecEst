"""Regression tests for tuple-valued mapping keys in CLI comparisons."""

from pyrecest.cli import _values_equal


def test_values_equal_preserves_tuple_mapping_keys():
    value = {("session", 1): {("target", 2): "matched"}}

    assert _values_equal(value, value)


def test_values_equal_detects_different_tuple_mapping_keys():
    actual = {("session", 1): "matched"}
    expected = {("session", 2): "matched"}

    assert not _values_equal(actual, expected)
