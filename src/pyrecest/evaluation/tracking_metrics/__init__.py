"""Generic HOTA, CLEAR, and identity tracking metrics.

Callers provide dense per-frame identities and pairwise similarities through
:class:`TrackingSequence`; parsing, geometry, and benchmark-specific reporting
remain outside this package.

The metric assignment flows are adapted from TrackEval under the following
license notice:

MIT License

Copyright (c) 2020 Jonathon Luiten

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from ._clear_identity import (
    ClearCounts,
    IdentityCounts,
    combine_clear,
    combine_identity,
    evaluate_clear,
    evaluate_identity,
    finalize_clear,
    finalize_identity,
)
from ._data import TrackingSequence
from ._hota import HOTA_ALPHAS, HotaCounts, combine_hota, evaluate_hota, finalize_hota

__all__ = [
    "HOTA_ALPHAS",
    "ClearCounts",
    "HotaCounts",
    "IdentityCounts",
    "TrackingSequence",
    "combine_clear",
    "combine_hota",
    "combine_identity",
    "evaluate_clear",
    "evaluate_hota",
    "evaluate_identity",
    "finalize_clear",
    "finalize_hota",
    "finalize_identity",
]
