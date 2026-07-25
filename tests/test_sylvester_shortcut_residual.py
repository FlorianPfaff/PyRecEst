import numpy as np
import numpy.testing as npt
from pyrecest._backend import numpy as numpy_backend


def test_numpy_sylvester_falls_back_when_close_factor_shortcut_is_inaccurate():
    a = np.diag([1e-8, 1.0])
    b = np.diag([2e-8, 1.0])
    q = np.eye(2)

    # The shortcut's approximate shared-factor check accepts this pair, but
    # replacing b by a changes the first solution entry by 50 percent.
    assert np.all(np.isclose(a, b))

    x = numpy_backend.linalg.solve_sylvester(a, b, q)

    residual = a @ x + x @ b - q
    npt.assert_allclose(residual, np.zeros_like(q), atol=1e-12)
    npt.assert_allclose(x[0, 0], 1.0 / 3e-8, rtol=1e-12)
