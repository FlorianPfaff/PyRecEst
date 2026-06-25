from types import SimpleNamespace

import numpy as np
from pyrecest.models.validation import infer_state_dim_from_distribution


def test_infer_state_dim_accepts_scalar_array_dim_attribute() -> None:
    distribution = SimpleNamespace(dim=np.array(4))

    assert infer_state_dim_from_distribution(distribution) == 4
