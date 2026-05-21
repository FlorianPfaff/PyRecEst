# API Stability And Deprecation Policy

PyRecEst can support both stable research workflows and fast-moving experimental
ideas by separating public APIs from experimental APIs.

## Stability Categories

| Category         | Meaning                                                                                              |
|------------------|------------------------------------------------------------------------------------------------------|
| Stable           | Public API covered by tests and deprecation policy.                                                  |
| Experimental     | API may change without the full deprecation cycle. Prefer `pyrecest.experimental` for these objects. |
| Deprecated       | API still exists but emits `DeprecationWarning` and has a planned removal version.                   |
| Backend-specific | API is stable only for the backends listed in the backend API matrix.                                |

## Deprecations

Use `pyrecest.deprecation.deprecated` for public API transitions:

```python
from pyrecest.deprecation import deprecated

@deprecated(since="2.3.0", remove_in="3.0.0", replacement="new_function")
def old_function():
    ...
```

Recommended cadence:

1. introduce the replacement and warning in a minor release;
2. keep the warning for at least one additional minor release;
3. remove only in a major release unless the API was explicitly experimental.
