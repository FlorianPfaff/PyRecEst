import runpy
from pathlib import Path

import numpy as np

d = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "track_evaluation.py"), run_name=__name__
)
o = d["_optional_int_candidate"]


def f(v):
    if isinstance(v, np.ndarray):
        if v.ndim != 0:
            return d["_MISSING"]
        v = v.item()
    if isinstance(v, (bool, np.bool_)):
        return d["_MISSING"]
    return o(v)


d["_optional_int_candidate"] = f
for n in d["__all__"]:
    globals()[n] = d[n]
__all__ = d["__all__"]
