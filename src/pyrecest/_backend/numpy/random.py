"""Numpy based random backend."""

import numpy as _np
from numpy.random import default_rng as _default_rng
from numpy.random import (  # For PyRecEst
    get_state,
    multinomial,
    randint,
    seed,
    set_state,
)

from .._shared_numpy.random import choice, multivariate_normal, normal, rand, uniform
