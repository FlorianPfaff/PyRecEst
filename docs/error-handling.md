# Error Handling

PyRecEst uses ordinary Python exceptions for validation and numerical failures.
New user-facing APIs should prefer the shared exception classes in
`pyrecest.exceptions` so backend limitations, shape issues, and numerical
instabilities are easier to diagnose.

## Shared Exceptions

| Exception                  | Use when                                                                                                                    |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `PyRecEstError`            | A caller wants to catch PyRecEst-specific failures as a group.                                                              |
| `BackendNotSupportedError` | An operation is unavailable for the active backend. This also subclasses `NotImplementedError`.                             |
| `ShapeError`               | An array, vector, matrix, particle set, or measurement set has the wrong shape.                                             |
| `DimensionMismatchError`   | Two objects have incompatible state, measurement, or ambient dimensions.                                                    |
| `NumericalStabilityError`  | A computation cannot be completed safely because of conditioning, definiteness, normalization, or related numerical issues. |

## Message Policy

Error messages should include enough context for a user to fix the problem
without inspecting internal code.  For new code, include:

- the public API or argument name;
- the active backend when backend behavior is relevant;
- the actual shape or dimension when validation fails;
- the expected shape, dimension, or backend support level;
- a short reason when the failure is numerical or intentionally unsupported.

## Backend Restrictions

Prefer explicit backend failures over silent conversion or partial behavior.  If
an operation is intentionally unavailable for a backend, use
`BackendNotSupportedError` or a clear `NotImplementedError`, add the limitation
to `src/pyrecest/_backend/capabilities.py`, and update the backend compatibility
docs when the affected API is user-facing.

## Numerical Failures

Numerical failures should be handled consistently:

1. Validate shapes and dimensions before expensive computations.
2. Check covariance symmetry and definiteness when an algorithm requires it.
3. Expose tolerances in tests and diagnostics when the tolerance is meaningful.
4. Raise `NumericalStabilityError` when continuing would produce misleading
   estimates, invalid probabilities, or invalid covariance matrices.

This policy is incremental.  Existing code may still raise `ValueError`,
`AssertionError`, or `NotImplementedError`; new code should use the shared
classes when the failure reaches users directly.
