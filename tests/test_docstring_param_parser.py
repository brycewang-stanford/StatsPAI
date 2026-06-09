"""Regression tests for ``registry._parse_docstring_params``.

The auto-registry derives parameter *descriptions / types / enums* from
docstrings (names come from the signature). The parser handles three real
dialects in the StatsPAI codebase:

* NumPy ``name : type`` headers,
* NumPy *type-less* ``name`` headers (description indented beneath) — the style
  ``feols`` / ``causal_forest`` / ``mixed`` use; these were silently dropped
  before the type-less fix, losing 65 param descriptions across 41 functions,
* Google ``name (type): desc`` headers.

These tests lock that behaviour and guard the false-positive boundary: an
indented description line must never be mistaken for a type-less header.
"""

import inspect
import textwrap

import statspai as sp
from statspai.registry import _parse_docstring_params


def _doc(s: str) -> str:
    """Mirror ``inspect.getdoc``: strip + dedent so names sit at column 0."""
    return textwrap.dedent(s).strip("\n")


def test_numpy_typed_headers():
    doc = _doc("""
    Parameters
    ----------
    data : DataFrame
        The input data.
    method : {"a", "b", "c"}
        Which method.
    """)
    p = _parse_docstring_params(doc)
    assert set(p) == {"data", "method"}
    assert p["data"]["type"] == "DataFrame"
    assert p["data"]["description"] == "The input data."
    assert p["method"]["enum"] == ["a", "b", "c"]


def test_numpy_typeless_headers():
    """Name on its own line, description indented — no ``: type``."""
    doc = _doc("""
    Parameters
    ----------
    data
        The input data.
    y
        Outcome column.
    """)
    p = _parse_docstring_params(doc)
    assert set(p) == {"data", "y"}
    assert p["data"]["description"] == "The input data."
    assert p["y"]["description"] == "Outcome column."


def test_typeless_comma_separated_names():
    """``restricted, full`` style (lrtest) shares one description block."""
    doc = _doc("""
    Parameters
    ----------
    restricted, full
        Two fitted models.
    """)
    p = _parse_docstring_params(doc)
    assert set(p) == {"restricted", "full"}
    assert p["restricted"]["description"] == "Two fitted models."
    assert p["full"]["description"] == "Two fitted models."


def test_google_headers():
    doc = _doc("""
    Args:
        data (DataFrame): The input data.
        k (int): Number of folds.
    """)
    p = _parse_docstring_params(doc)
    assert set(p) == {"data", "k"}
    assert p["data"]["description"] == "The input data."
    assert p["k"]["type"] == "int"


def test_indented_word_is_not_a_typeless_header():
    """A one-word *description* line must not be parsed as a new param."""
    doc = _doc("""
    Parameters
    ----------
    data : DataFrame
        Input.
        Continued.
    """)
    p = _parse_docstring_params(doc)
    # 'Continued.' is an indented continuation, NOT a new 'Continued' param.
    assert set(p) == {"data"}
    assert "Continued" not in p


def test_section_terminates_at_returns():
    doc = _doc("""
    Parameters
    ----------
    x
        A value.

    Returns
    -------
    result
        Not a parameter.
    """)
    p = _parse_docstring_params(doc)
    assert set(p) == {"x"}
    assert "result" not in p


def test_real_functions_recover_typeless_params():
    """End-to-end: previously-dropped type-less params are now extracted."""
    for name, expected in [
        ("mixed", {"data", "y", "group"}),
        ("lrtest", {"restricted", "full"}),
        ("evalue_rr", {"rr"}),
    ]:
        p = _parse_docstring_params(inspect.getdoc(getattr(sp, name)) or "")
        assert expected.issubset(set(p)), f"{name}: missing {expected - set(p)}"
