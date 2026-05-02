"""Tests for public conversion capability protocols."""

from __future__ import annotations

from pyrecest.distributions.abstract_distribution_type import AbstractDistributionType
from pyrecest.distributions.conversion import ConversionResult
from pyrecest.protocols.conversions import (
    ConversionAliasResolver,
    ConversionParameters,
    ConversionTarget,
    DistributionConverter,
    SupportsApproximateAs,
    SupportsConvertTo,
    SupportsDistributionConversion,
    SupportsFromDistribution,
)


class SourceDistribution(AbstractDistributionType):
    """Minimal source distribution with PyRecEst conversion wrappers."""


class TargetDistribution:
    """Minimal target representation using target-centric construction."""

    def __init__(self, source, parameters):
        self.source = source
        self.parameters = parameters

    @classmethod
    def from_distribution(cls, distribution, /, **kwargs):
        return cls(distribution, dict(kwargs))


def convert_to_target(distribution, /, **kwargs):
    return TargetDistribution.from_distribution(distribution, **kwargs)


def resolve_target(distribution, /):
    if distribution is None:
        raise TypeError("distribution must not be None")
    return TargetDistribution


def test_conversion_protocols_are_importable():
    assert ConversionTarget is not None
    assert ConversionParameters is not None
    assert SupportsFromDistribution is not None
    assert SupportsConvertTo is not None
    assert SupportsApproximateAs is not None
    assert SupportsDistributionConversion is not None


def test_from_distribution_protocol_is_runtime_checkable_on_target_class():
    assert isinstance(TargetDistribution, SupportsFromDistribution)
    assert issubclass(TargetDistribution, SupportsFromDistribution)


def test_distribution_conversion_wrappers_are_runtime_checkable():
    source = SourceDistribution()

    assert isinstance(source, SupportsConvertTo)
    assert isinstance(source, SupportsApproximateAs)
    assert isinstance(source, SupportsDistributionConversion)


def test_registered_conversion_callable_protocols_are_runtime_checkable():
    source = SourceDistribution()

    assert isinstance(convert_to_target, DistributionConverter)
    assert isinstance(resolve_target, ConversionAliasResolver)
    assert convert_to_target(source, scale=2).parameters == {"scale": 2}
    assert resolve_target(source) is TargetDistribution


def test_conversion_wrapper_protocol_matches_existing_gateway_behavior():
    source = SourceDistribution()

    converted = source.convert_to(TargetDistribution, scale=2)

    assert isinstance(converted, TargetDistribution)
    assert converted.source is source
    assert converted.parameters == {"scale": 2}


def test_approximate_as_protocol_supports_return_info():
    source = SourceDistribution()

    result = source.approximate_as(TargetDistribution, return_info=True, scale=3)

    assert isinstance(result, ConversionResult)
    assert isinstance(result.distribution, TargetDistribution)
    assert result.distribution.source is source
    assert result.distribution.parameters == {"scale": 3}
    assert result.target_type is TargetDistribution
    assert result.exact is False
