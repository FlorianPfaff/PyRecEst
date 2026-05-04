"""Import compatibility checks for the public protocol rollout."""

from __future__ import annotations

from pyrecest.models import SupportsLikelihood as ModelSupportsLikelihood
from pyrecest.models import SupportsLogLikelihood as ModelSupportsLogLikelihood
from pyrecest.models import SupportsTransitionDensity as ModelSupportsTransitionDensity
from pyrecest.models import (
    SupportsTransitionSampling as ModelSupportsTransitionSampling,
)
from pyrecest.models.likelihood import (
    SupportsLikelihood as LikelihoodSupportsLikelihood,
)
from pyrecest.models.likelihood import (
    SupportsLogLikelihood as LikelihoodSupportsLogLikelihood,
)
from pyrecest.models.likelihood import (
    SupportsTransitionDensity as LikelihoodSupportsTransitionDensity,
)
from pyrecest.models.likelihood import (
    SupportsTransitionSampling as LikelihoodSupportsTransitionSampling,
)
from pyrecest.protocols import SupportsDim as PackageSupportsDim
from pyrecest.protocols.common import SupportsDim as CommonSupportsDim


def test_common_protocol_package_import_remains_available():
    assert PackageSupportsDim is CommonSupportsDim


def test_existing_model_protocol_imports_remain_available():
    assert ModelSupportsLikelihood is LikelihoodSupportsLikelihood
    assert ModelSupportsLogLikelihood is LikelihoodSupportsLogLikelihood
    assert ModelSupportsTransitionDensity is LikelihoodSupportsTransitionDensity
    assert ModelSupportsTransitionSampling is LikelihoodSupportsTransitionSampling
