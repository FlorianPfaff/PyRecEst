"""Backend support matrix data shared by tests and documentation tooling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapability:
    """One documented backend capability row."""

    area: str
    capability: str
    numpy: str
    pytorch: str
    jax: str
    notes: str = ""


CAPABILITIES: tuple[BackendCapability, ...] = (
    BackendCapability(
        "backend.random",
        "Seeded scalar/vector normal sampling",
        "yes",
        "yes",
        "yes",
        "JAX uses a process-global PRNG key unless explicit state is passed.",
    ),
    BackendCapability(
        "backend.random",
        "Weighted choice with replacement",
        "yes",
        "yes",
        "partial",
        "JAX support depends on argument form and should be covered by focused tests.",
    ),
    BackendCapability(
        "backend.random",
        "Weighted choice without replacement",
        "yes",
        "no",
        "partial",
        "PyTorch backend intentionally rejects weighted sampling without replacement.",
    ),
    BackendCapability(
        "distributions",
        "GaussianDistribution.pdf / ln_pdf",
        "yes",
        "yes",
        "yes",
        "Smoke-tested with reference values.",
    ),
    BackendCapability(
        "filters",
        "KalmanFilter.predict_linear / update_linear",
        "yes",
        "yes",
        "yes",
        "Backend-portable linear algebra path.",
    ),
    BackendCapability(
        "filters",
        "UKFOnManifolds.predict / update",
        "yes",
        "yes",
        "no",
        "JAX is explicitly rejected by this API.",
    ),
    BackendCapability(
        "utilities",
        "SciPy-heavy tracking/evaluation helpers",
        "yes",
        "partial",
        "partial",
        "Check NumPy behavior first for advanced workflows.",
    ),
)


def markdown_table() -> str:
    """Return the support matrix as a Markdown table."""
    lines = [
        "| Area | Capability | NumPy | PyTorch | JAX | Notes |",
        "|------|------------|:-----:|:-------:|:---:|-------|",
    ]
    for capability in CAPABILITIES:
        lines.append(
            "| {area} | {capability} | {numpy} | {pytorch} | {jax} | {notes} |".format(
                area=capability.area,
                capability=capability.capability,
                numpy=capability.numpy,
                pytorch=capability.pytorch,
                jax=capability.jax,
                notes=capability.notes,
            )
        )
    return "\n".join(lines)
