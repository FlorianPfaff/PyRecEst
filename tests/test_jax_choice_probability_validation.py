import importlib.util
import textwrap
import unittest

from tests.support.backend_runner import run_backend_code


@unittest.skipIf(importlib.util.find_spec("jax") is None, "JAX is not installed")
class JaxChoiceProbabilityValidationTest(unittest.TestCase):
    def assert_choice_raises_value_error(self, call_source):
        code = textwrap.dedent(f"""
            from pyrecest.backend import array, random

            try:
                {call_source}
            except ValueError:
                raise SystemExit(0)
            except Exception as exc:
                raise AssertionError(
                    f"expected ValueError, got {{type(exc).__name__}}: {{exc}}"
                ) from exc
            else:
                raise AssertionError("expected ValueError")
            """)
        result = run_backend_code("jax", code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_choice_rejects_negative_probabilities(self):
        self.assert_choice_raises_value_error(
            "random.choice(array([0, 1, 2]), size=2, p=array([0.5, -0.1, 0.6]))"
        )

    def test_choice_rejects_non_finite_probabilities(self):
        self.assert_choice_raises_value_error(
            "random.choice(array([0, 1, 2]), size=2, p=array([0.5, float('nan'), 0.5]))"
        )

    def test_choice_rejects_zero_sum_probabilities(self):
        self.assert_choice_raises_value_error(
            "random.choice(array([0, 1, 2]), size=2, p=array([0.0, 0.0, 0.0]))"
        )

    def test_choice_rejects_wrong_probability_length(self):
        self.assert_choice_raises_value_error(
            "random.choice(array([0, 1, 2]), size=2, p=array([0.5, 0.5]))"
        )


if __name__ == "__main__":
    unittest.main()
