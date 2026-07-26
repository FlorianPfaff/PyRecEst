"""Regression tests for circular association likelihood validation."""

import math

import pytest
from pyrecest.backend import array
from pyrecest.filters.abstract_circular_filter import AbstractCircularFilter


class _TestCircularFilter(AbstractCircularFilter):
    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        self._filter_state = new_state


class _PdfDistribution:
    def __init__(self, pdf_value):
        self._pdf_value = pdf_value

    def pdf(self, _angle):
        return self._pdf_value

    def mean(self):
        return array([0.0])

    @property
    def dim(self):
        return 1


def test_association_likelihood_accepts_single_element_pdf_values():
    filt = _TestCircularFilter(_PdfDistribution(array([1.0])))
    likelihood = _PdfDistribution(array(0.5))

    result = filt.association_likelihood_numerical(likelihood)

    assert float(result) == pytest.approx(0.5 * math.tau)


@pytest.mark.parametrize(
    ("state_pdf", "likelihood_pdf", "message"),
    [
        (array([1.0, 2.0]), array([1.0]), r"filter_state\.pdf"),
        (array([1.0]), array([1.0, 2.0]), r"likelihood\.pdf"),
    ],
)
def test_association_likelihood_rejects_vector_pdf_values(
    state_pdf,
    likelihood_pdf,
    message,
):
    filt = _TestCircularFilter(_PdfDistribution(state_pdf))
    likelihood = _PdfDistribution(likelihood_pdf)

    with pytest.raises(ValueError, match=message):
        filt.association_likelihood_numerical(likelihood)


@pytest.mark.parametrize(
    ("state_pdf", "likelihood_pdf", "message"),
    [
        (array([float("nan")]), array([1.0]), r"filter_state\.pdf"),
        (array([float("inf")]), array([1.0]), r"filter_state\.pdf"),
        (array([1.0]), array([float("nan")]), r"likelihood\.pdf"),
        (array([1.0]), array([-float("inf")]), r"likelihood\.pdf"),
    ],
)
def test_association_likelihood_rejects_nonfinite_pdf_values(
    state_pdf,
    likelihood_pdf,
    message,
):
    filt = _TestCircularFilter(_PdfDistribution(state_pdf))
    likelihood = _PdfDistribution(likelihood_pdf)

    with pytest.raises(ValueError, match=message):
        filt.association_likelihood_numerical(likelihood)
