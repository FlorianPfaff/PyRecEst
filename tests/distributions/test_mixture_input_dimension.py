from types import SimpleNamespace

import pytest
from pyrecest.backend import array
from pyrecest.distributions.abstract_mixture import AbstractMixture


def test_mixture_rejects_components_with_different_input_dimensions():
    distributions = [
        SimpleNamespace(dim=2, input_dim=2),
        SimpleNamespace(dim=2, input_dim=3),
    ]

    with pytest.raises(
        ValueError,
        match="All distributions must have the same input dimension",
    ):
        AbstractMixture(distributions, array([0.5, 0.5]))


def test_mixture_accepts_components_with_matching_dimensions():
    distributions = [
        SimpleNamespace(dim=2, input_dim=3),
        SimpleNamespace(dim=2, input_dim=3),
    ]

    mixture = AbstractMixture(distributions, array([0.5, 0.5]))

    assert mixture.input_dim == 3
