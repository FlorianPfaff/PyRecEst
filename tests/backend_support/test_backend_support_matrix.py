from tools.backend_support_matrix import CAPABILITIES, markdown_table


def test_backend_support_matrix_has_expected_columns():
    assert CAPABILITIES
    for capability in CAPABILITIES:
        assert capability.area
        assert capability.capability
        assert capability.numpy in {"yes", "no", "partial"}
        assert capability.pytorch in {"yes", "no", "partial"}
        assert capability.jax in {"yes", "no", "partial"}


def test_backend_support_markdown_table_contains_capabilities():
    table = markdown_table()
    assert "| Area | Capability | NumPy | PyTorch | JAX | Notes |" in table
    assert "GaussianDistribution.pdf / ln_pdf" in table
