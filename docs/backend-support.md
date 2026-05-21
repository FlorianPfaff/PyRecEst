# Backend Support Matrix

This page is generated from `tools/backend_support_matrix.py`. It records a
smoke-test-backed support snapshot for selected public APIs. It is intentionally
conservative and does not prove full mathematical or numerical parity.

| Area           | Capability                                  | NumPy | PyTorch |   JAX   | Notes                                                                        |
|----------------|---------------------------------------------|:-----:|:-------:|:-------:|------------------------------------------------------------------------------|
| backend.random | Seeded scalar/vector normal sampling        |  yes  |   yes   |   yes   | JAX uses a process-global PRNG key unless explicit state is passed.          |
| backend.random | Weighted choice with replacement            |  yes  |   yes   | partial | JAX support depends on argument form and should be covered by focused tests. |
| backend.random | Weighted choice without replacement         |  yes  |   no    | partial | PyTorch backend intentionally rejects weighted sampling without replacement. |
| distributions  | GaussianDistribution.pdf / ln_pdf           |  yes  |   yes   |   yes   | Smoke-tested with reference values.                                          |
| filters        | KalmanFilter.predict_linear / update_linear |  yes  |   yes   |   yes   | Backend-portable linear algebra path.                                        |
| filters        | UKFOnManifolds.predict / update             |  yes  |   yes   |   no    | JAX is explicitly rejected by this API.                                      |
| utilities      | SciPy-heavy tracking/evaluation helpers     |  yes  | partial | partial | Check NumPy behavior first for advanced workflows.                           |
