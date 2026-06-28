import numpy as np
import pytest
from pyrecest._backend.numpy import random


def test_multinomial_rejects_boolean_pvals():
    with pytest.raises(TypeError, match="real numeric"):
        random.multinomial(1, np.ones(1, dtype=bool))


def test_multinomial_rejects_text_pvals():
    with pytest.raises(TypeError, match="real numeric"):
        random.multinomial(1, ["1.0"])


def test_multinomial_accepts_real_pvals():
    result = random.multinomial(1, [1.0])

    assert result.shape == (1,)
    assert result.sum() == 1
