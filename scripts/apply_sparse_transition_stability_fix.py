from pathlib import Path

SOURCE_PATH = Path("src/pyrecest/filters/discrete_state.py")
TEST_PATH = Path("tests/filters/test_discrete_state_grid_validation.py")

OLD_BLOCK = '''    radius2 = np.inf if np.isinf(max_step_sigma) else (sigma * max_step_sigma) ** 2

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(states):
        delta = states - center[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius2
        if valid_mask is not None:
            keep &= valid_mask
        if not np.any(keep):
            keep[int(allowed[int(np.argmin(dist2[allowed]))])] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma * sigma))
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.zeros_like(weights)
            weights[int(np.argmin(dist2[dst]))] = 1.0
        else:
            weights /= weight_sum
        rows.extend(int(index) for index in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
'''

NEW_BLOCK = '''    max_weight_distance = float(
        np.sqrt(-2.0 * np.log(np.nextafter(0.0, 1.0)))
    )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(states):
        center_row = center[None, :]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled_delta = (states - center_row) / sigma
        unstable = ~np.isfinite(scaled_delta)
        if np.any(unstable):
            scale = np.maximum(
                np.maximum(np.abs(states), np.abs(center_row)),
                sigma,
            )
            with np.errstate(
                over="ignore",
                invalid="ignore",
                divide="ignore",
                under="ignore",
            ):
                fallback_numerator = states / scale - center_row / scale
                fallback_denominator = sigma / scale
                fallback = np.divide(
                    fallback_numerator,
                    fallback_denominator,
                    out=np.full_like(fallback_numerator, np.inf),
                    where=fallback_denominator > 0.0,
                )
            fallback[states == center_row] = 0.0
            scaled_delta[unstable] = fallback[unstable]
        distances = np.hypot.reduce(scaled_delta, axis=1)
        keep = distances <= max_step_sigma
        if valid_mask is not None:
            keep &= valid_mask
        if not np.any(keep):
            keep[int(allowed[int(np.argmin(distances[allowed]))])] = True
        dst = np.flatnonzero(keep)
        destination_distances = distances[dst]
        weights = np.zeros_like(destination_distances)
        representable = destination_distances <= max_weight_distance
        with np.errstate(under="ignore"):
            weights[representable] = np.exp(
                -0.5 * destination_distances[representable] ** 2
            )
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.zeros_like(weights)
            weights[int(np.argmin(destination_distances))] = 1.0
        else:
            weights /= weight_sum
        rows.extend(int(index) for index in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
'''

TEST_METHOD = '''
    def test_sparse_gaussian_transition_matrix_handles_extreme_finite_scale(self):
        states = np.array([-1.0e308, 1.0e308])

        with np.errstate(over="raise", invalid="raise"):
            transition = sparse_gaussian_transition_matrix(
                states,
                sigma=1.0e308,
                max_step_sigma=np.inf,
            ).toarray()

        cross_weight = np.exp(-2.0)
        expected = np.array(
            [[1.0, cross_weight], [cross_weight, 1.0]],
            dtype=float,
        ) / (1.0 + cross_weight)
        self.assertTrue(np.isfinite(transition).all())
        np.testing.assert_allclose(transition, expected, rtol=1.0e-14, atol=0.0)
        np.testing.assert_allclose(transition.sum(axis=0), np.ones(2))
'''


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    if source.count(OLD_BLOCK) != 1:
        raise RuntimeError("Expected exactly one sparse transition implementation block")
    SOURCE_PATH.write_text(source.replace(OLD_BLOCK, NEW_BLOCK), encoding="utf-8")

    tests = TEST_PATH.read_text(encoding="utf-8")
    marker = '\n\nif __name__ == "__main__":\n'
    if tests.count(marker) != 1:
        raise RuntimeError("Could not locate test module footer")
    TEST_PATH.write_text(tests.replace(marker, TEST_METHOD + marker), encoding="utf-8")


if __name__ == "__main__":
    main()
