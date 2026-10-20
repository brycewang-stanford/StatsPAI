"""MR-PRESSO moved from ``mendelian.diagnostics`` to ``mendelian.presso``.

The old import path must keep resolving to the very objects that
``sp.mr_presso`` and the dispatcher use.
"""

from __future__ import annotations

import pytest

import statspai as sp
from statspai.mendelian import diagnostics, presso


@pytest.mark.parametrize(
    "name",
    [
        "mr_presso",
        "MRPressoResult",
        "_ivw_regression",
        "_loo_slopes",
        "_presso_rss",
        "_distortion_pvalue",
    ],
)
def test_old_import_path_resolves_to_the_new_module(name):
    assert getattr(diagnostics, name) is getattr(presso, name)


def test_public_function_is_the_same_object_everywhere():
    # sp.mr_presso is a lazy export: the first access wraps it and rebinds
    # the wrapper in its defining module (as before the move).
    public = sp.mr_presso
    from statspai.mendelian.diagnostics import mr_presso

    assert mr_presso is public is presso.mr_presso
    assert "mr_presso" in diagnostics.__all__


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        diagnostics.not_a_function  # noqa: B018
