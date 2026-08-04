from pathlib import Path
from textwrap import dedent


METRICS_PATH = Path("src/pyrecest/utils/metrics.py")
TEST_PATH = Path("tests/test_metrics_covariance_validation.py")

OLD_BLOCK = '''def _as_covariance_stack(
    covariances: ArrayLike, n_samples: int, dim: int, name: str
) -> np.ndarray:
    covariance_array = _as_numeric_array(covariances, name)
    if covariance_array.ndim == 2:
        if covariance_array.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        return np.broadcast_to(covariance_array, (n_samples, dim, dim))
    if covariance_array.ndim == 3:
        if covariance_array.shape != (n_samples, dim, dim):
            raise ValueError(f"{name} must have shape ({n_samples}, {dim}, {dim})")
        return covariance_array
    raise ValueError(f"{name} must have shape (dim, dim) or (n, dim, dim)")
'''

NEW_BLOCK = '''def _as_covariance_stack(
    covariances: ArrayLike, n_samples: int, dim: int, name: str
) -> np.ndarray:
    covariance_array = _as_numeric_array(covariances, name)
    if covariance_array.ndim == 2:
        if covariance_array.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        covariance_stack = np.broadcast_to(
            covariance_array, (n_samples, dim, dim)
        )
    elif covariance_array.ndim == 3:
        if covariance_array.shape != (n_samples, dim, dim):
            raise ValueError(f"{name} must have shape ({n_samples}, {dim}, {dim})")
        covariance_stack = covariance_array
    else:
        raise ValueError(f"{name} must have shape (dim, dim) or (n, dim, dim)")

    if not np.all(np.isfinite(covariance_stack)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(
        covariance_stack,
        np.swapaxes(covariance_stack, -1, -2),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(f"{name} must contain symmetric covariance matrices")
    try:
        np.linalg.cholesky(covariance_stack)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{name} must contain positive-definite covariance matrices"
        ) from exc
    return covariance_stack
'''

TEST_CONTENT = dedent(
    '''\
    import unittest

    import numpy as np
    import numpy.testing as npt

    from pyrecest.utils.metrics import nees, nis


    class TestConsistencyMetricCovarianceValidation(unittest.TestCase):
        def test_nees_and_nis_reject_asymmetric_covariances(self):
            residual = np.array([1.0, 1.0])
            asymmetric_covariance = np.array([[1.0, 10.0], [0.0, 1.0]])

            with self.assertRaisesRegex(
                ValueError,
                "uncertainties must contain symmetric covariance matrices",
            ):
                nees(residual, asymmetric_covariance)

            with self.assertRaisesRegex(
                ValueError,
                "innovation_covariances must contain symmetric covariance matrices",
            ):
                nis(residual, asymmetric_covariance)

        def test_nees_and_nis_reject_non_positive_definite_covariances(self):
            residuals = np.array([[1.0, 0.0], [0.0, 1.0]])
            covariance_stack = np.array(
                [
                    np.eye(2),
                    [[1.0, 2.0], [2.0, 1.0]],
                ]
            )

            with self.assertRaisesRegex(
                ValueError,
                "uncertainties must contain positive-definite covariance matrices",
            ):
                nees(residuals, covariance_stack)

            with self.assertRaisesRegex(
                ValueError,
                "innovation_covariances must contain positive-definite covariance matrices",
            ):
                nis(residuals, covariance_stack)

        def test_valid_covariance_stacks_remain_supported(self):
            residuals = np.array([[1.0, 0.0], [0.0, 2.0]])
            covariance_stack = np.array(
                [
                    np.eye(2),
                    [[2.0, 0.0], [0.0, 4.0]],
                ]
            )

            npt.assert_allclose(nees(residuals, covariance_stack), [1.0, 1.0])
            npt.assert_allclose(nis(residuals, covariance_stack), [1.0, 1.0])


    if __name__ == "__main__":
        unittest.main()
    '''
)


def main() -> None:
    text = METRICS_PATH.read_text()
    if NEW_BLOCK in text:
        raise SystemExit("covariance validation is already installed")
    if OLD_BLOCK not in text:
        raise SystemExit("expected covariance-stack implementation not found")
    METRICS_PATH.write_text(text.replace(OLD_BLOCK, NEW_BLOCK))
    TEST_PATH.write_text(TEST_CONTENT)


if __name__ == "__main__":
    main()
