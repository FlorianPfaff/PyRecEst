"""Autograd based random backend."""

import autograd.numpy as _np
from autograd.numpy.random import get_state, multinomial, randint, seed, set_state

from .._shared_numpy.random import choice, multivariate_normal, normal, rand, uniform
