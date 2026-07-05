import os

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
