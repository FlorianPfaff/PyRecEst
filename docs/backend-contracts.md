# Backend Contract Testing

PyRecEst has a dynamic backend facade, so backend support needs executable
contracts in addition to prose documentation.

| Contract surface  | Check                                                                                  |
|-------------------|----------------------------------------------------------------------------------------|
| Facade metadata   | Every declared unsupported or partial function must exist on the active facade module. |
| Public API matrix | Every API row must name NumPy, PyTorch, JAX, and explanatory notes.                    |
| Portable examples | Core examples should run under each backend they claim to support.                     |
| Unsupported paths | Unsupported backend paths should raise clear backend-named errors.                     |

When adding a backend-specific restriction, update
`src/pyrecest/_backend/capabilities.py`, the backend API matrix, and a focused
regression test in the same change.
