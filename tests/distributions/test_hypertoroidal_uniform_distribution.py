import os
import subprocess
import sys

import pytest
from pyrecest.backend import array, ones, pi, zeros
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from pyrecest.exceptions import ShapeError


def test_pdf_validates_dimension_mismatch():
    dist = HypertoroidalUniformDistribution(2)

    with pytest.raises(ShapeError, match="xs"):
        dist.pdf(array([0.1]))


def test_pdf_rejects_scalar_for_multidimensional_distribution():
    dist = HypertoroidalUniformDistribution(2)

    with pytest.raises(ShapeError, match="scalar inputs"):
        dist.pdf(array(0.1))


def test_pdf_keeps_legacy_shape_for_valid_inputs():
    dist = HypertoroidalUniformDistribution(2)

    values = dist.pdf(array([[0.1, 0.2], [0.3, 0.4]]))

    assert values.shape == (2,)


def test_shift_validates_shape():
    dist = HypertoroidalUniformDistribution(2)

    with pytest.raises(ShapeError, match="shift_by"):
        dist.shift(array([0.1]))


def test_integrate_validates_boundary_shapes():
    dist = HypertoroidalUniformDistribution(2)

    with pytest.raises(ShapeError, match="left"):
        dist.integrate((zeros((1,)), ones((2,))))

    with pytest.raises(ShapeError, match="right"):
        dist.integrate((zeros((2,)), ones((1,))))


def test_integrate_accepts_scalar_boundaries_for_one_dimension():
    dist = HypertoroidalUniformDistribution(1)

    assert dist.integrate((array(0.0), 2.0 * pi)) == pytest.approx(1.0)


def test_validation_survives_optimized_python():
    env = os.environ.copy()
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
from pyrecest.backend import array
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from pyrecest.exceptions import ShapeError

dist = HypertoroidalUniformDistribution(2)
for operation in (
    lambda: dist.pdf(array([0.1])),
    lambda: dist.shift(array([0.1])),
    lambda: dist.integrate((array([0.0]), array([1.0, 1.0]))),
):
    try:
        operation()
    except ShapeError:
        pass
    else:
        raise AssertionError("invalid shape was accepted under optimized Python")
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True, env=env)
