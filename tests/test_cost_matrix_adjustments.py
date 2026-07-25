import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.cost_matrix_adjustments import (
    CallableCostMatrixAdjustment,
    CostMatrixAdjustmentResult,
    additive_cost_matrix_adjustment,
    apply_cost_matrix_adjustment,
    compose_cost_matrix_adjustments,
)


class _ObjectAdjustment:
    name = "object_adjustment"

    def apply(self, cost_matrix, *, metadata=None):
        scale = float((metadata or {}).get("scale", 1.0))
        return CostMatrixAdjustmentResult(
            np.asarray(cost_matrix, dtype=float) + scale,
            {"scale": scale},
        )


class TestCostMatrixAdjustments(unittest.TestCase):
    def test_callable_adjustment_returns_matrix_and_preserves_shape(self):
        adjustment = CallableCostMatrixAdjustment(
            name="add_one",
            function=lambda matrix, metadata: matrix + float(metadata["offset"]),
            metadata={"offset": 1.0},
        )

        result = apply_cost_matrix_adjustment(
            np.array([[1.0, 2.0], [3.0, np.inf]]),
            adjustment,
        )

        npt.assert_allclose(
            result.adjusted_cost_matrix,
            np.array([[2.0, 3.0], [4.0, np.inf]]),
        )
        self.assertEqual(dict(result.diagnostics), {})

    def test_callable_adjustment_metadata_override_and_diagnostics_tuple(self):
        def adjustment(matrix, metadata):
            return matrix * float(metadata["scale"]), {"used_scale": metadata["scale"]}

        wrapped = CallableCostMatrixAdjustment(
            name="scale",
            function=adjustment,
            metadata={"scale": 2.0},
        )

        result = wrapped.apply(np.array([[1.0]]), metadata={"scale": 3.0})

        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[3.0]]))
        self.assertEqual(dict(result.diagnostics), {"used_scale": 3.0})

    def test_object_adjustment_receives_metadata(self):
        result = apply_cost_matrix_adjustment(
            np.array([[0.0, 1.0]]),
            _ObjectAdjustment(),
            metadata={"scale": 2.5},
        )

        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[2.5, 3.5]]))
        self.assertEqual(dict(result.diagnostics), {"scale": 2.5})

    def test_compose_adjustments_collects_ordered_diagnostics(self):
        add = additive_cost_matrix_adjustment(
            np.array([[1.0, 0.0]]),
            name="prior",
            diagnostics={"kind": "prior"},
        )
        scale = CallableCostMatrixAdjustment(
            name="scale",
            function=lambda matrix, _metadata: CostMatrixAdjustmentResult(
                matrix * 2.0,
                {"factor": 2.0},
            ),
        )

        result = compose_cost_matrix_adjustments(
            np.array([[2.0, 3.0]]),
            [add, scale],
        )

        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[6.0, 6.0]]))
        self.assertEqual(result.diagnostics["adjustment_order"], ["prior", "scale"])
        self.assertEqual(result.diagnostics["prior"], {"kind": "prior"})
        self.assertEqual(result.diagnostics["scale"], {"factor": 2.0})

    def test_compose_adjustments_disambiguates_duplicate_diagnostics(self):
        first = CallableCostMatrixAdjustment(
            name="shared",
            function=lambda matrix, _metadata: CostMatrixAdjustmentResult(
                matrix + 1.0,
                {"step": 1},
            ),
        )
        second = CallableCostMatrixAdjustment(
            name="shared",
            function=lambda matrix, _metadata: CostMatrixAdjustmentResult(
                matrix + 2.0,
                {"step": 2},
            ),
        )

        result = compose_cost_matrix_adjustments(np.array([[0.0]]), [first, second])

        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[3.0]]))
        self.assertEqual(result.diagnostics["adjustment_order"], ["shared", "shared_2"])
        self.assertEqual(result.diagnostics["shared"], {"step": 1})
        self.assertEqual(result.diagnostics["shared_2"], {"step": 2})

    def test_additive_adjustment_rejects_shape_mismatch(self):
        adjustment = additive_cost_matrix_adjustment(np.ones((2, 2)))

        with self.assertRaisesRegex(ValueError, "penalty_matrix shape"):
            apply_cost_matrix_adjustment(np.ones((1, 2)), adjustment)

    def test_adjustment_rejects_shape_change(self):
        adjustment = CallableCostMatrixAdjustment(
            name="bad",
            function=lambda _matrix, _metadata: np.ones((1, 3)),
        )

        with self.assertRaisesRegex(ValueError, r"expected \(1, 2\)"):
            apply_cost_matrix_adjustment(np.ones((1, 2)), adjustment)

    def test_simple_callable_without_apply_is_supported(self):
        result = apply_cost_matrix_adjustment(
            np.array([[1.0, 2.0]]),
            lambda matrix: matrix + 4.0,
        )

        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[5.0, 6.0]]))

    def test_numeric_validation(self):
        invalid_inputs = (
            [[True]],
            [["not-a-number"]],
            [1.0, 2.0],
            [[-np.inf]],
            [[np.complex64(1.0 + 0.0j)]],
        )
        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    apply_cost_matrix_adjustment(value, lambda matrix: matrix)

    def test_named_adjustment_validation(self):
        with self.assertRaises(ValueError):
            CallableCostMatrixAdjustment(
                name="", function=lambda matrix, metadata: matrix
            )
        with self.assertRaises(ValueError):
            CallableCostMatrixAdjustment(name="bad", function=None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
