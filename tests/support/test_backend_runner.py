import os
from pathlib import Path

import pytest

from tests.support import backend_runner


def test_subprocess_pythonpath_prefers_checkout_src():
    existing = os.pathsep.join(["/tmp/site-packages", "/tmp/other"])

    pythonpath = backend_runner._subprocess_pythonpath(existing)

    paths = pythonpath.split(os.pathsep)
    assert paths[0] == str(backend_runner._REPO_ROOT / "src")
    assert paths[1:] == existing.split(os.pathsep)


def test_subprocess_pythonpath_does_not_duplicate_checkout_src():
    src_path = str(backend_runner._REPO_ROOT / "src")
    existing = os.pathsep.join([src_path, "/tmp/site-packages"])

    pythonpath = backend_runner._subprocess_pythonpath(existing)

    assert pythonpath == existing


@pytest.mark.parametrize("existing", [None, ""])
def test_subprocess_pythonpath_handles_unset_or_empty_value(existing):
    assert backend_runner._subprocess_pythonpath(existing) == backend_runner._SRC_PATH


@pytest.mark.parametrize(
    ("parts", "expected_tail"),
    [
        (["older", "SRC"], ["older"]),
        (["older", "SRC", "other"], ["older", "other"]),
        (["SRC", "older", "SRC"], ["older"]),
        (["older", "SRC", "SRC", "other"], ["older", "other"]),
        (["", "older", "SRC", "older", ""], ["", "older", "older", ""]),
    ],
)
def test_subprocess_pythonpath_moves_only_checkout_src_to_front(parts, expected_tail):
    src_path = backend_runner._SRC_PATH
    existing = os.pathsep.join(src_path if part == "SRC" else part for part in parts)

    pythonpath = backend_runner._subprocess_pythonpath(existing)

    assert pythonpath.split(os.pathsep) == [src_path, *expected_tail]


def test_backend_subprocess_imports_checkout_before_older_installation(
    tmp_path, monkeypatch
):
    older_installation = tmp_path / "older-installation"
    older_package = older_installation / "pyrecest"
    older_package.mkdir(parents=True)
    (older_package / "__init__.py").write_text(
        'raise RuntimeError("older installation imported")\n', encoding="utf-8"
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(older_installation), backend_runner._SRC_PATH]),
    )
    monkeypatch.chdir(tmp_path)
    expected_source = Path(backend_runner._SRC_PATH) / "pyrecest" / "__init__.py"
    code = f"""
from pathlib import Path
import pyrecest
assert Path(pyrecest.__file__).resolve() == Path({str(expected_source)!r}).resolve()
"""

    result = backend_runner.run_backend_code("numpy", code, timeout=60)

    assert result.returncode == 0, result.stdout + result.stderr
