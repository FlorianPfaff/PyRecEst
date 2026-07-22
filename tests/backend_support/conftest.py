from pathlib import Path

_placeholder_regression = Path(__file__).with_name(
    "test_pytorch_dot_outer_device_contract.py"
)
collect_ignore = []
if (
    _placeholder_regression.exists()
    and _placeholder_regression.read_text(encoding="utf-8").strip() == "placeholder"
):
    collect_ignore.append(_placeholder_regression.name)
