"""Shared agent-native result protocol for domain result dataclasses.

Design principle 3 ("统一结果对象") asks every result object to be usable
through one entry point by a human *and* an agent. The flagship
`CausalResult` / `EconometricResults` already expose `.to_dict()` /
`.to_latex()` / `.cite()`; the lighter domain result dataclasses (negative
controls, proximal, four-way mediation, network spillover, ITS, Rosenbaum
bounds, longitudinal TMLE, QTE, robustness, transport, … and the BCF / MR
extensions) historically only had `.summary()`.

`ResultProtocolMixin` gives any such dataclass the three methods in one line of
inheritance, reusing the existing `core.results._to_jsonable` converter so the
serialisation logic lives in exactly one place (CLAUDE.md §4):

* ``to_dict()``  — JSON-safe ``{field: value}`` (numpy / pandas / NaN aware).
* ``to_latex()`` — a compact ``booktabs`` table of the scalar fields.
* ``cite()``     — the **verified** paper.bib key(s) the estimator is based on
  (set via the ``_citation_keys`` class attribute), or an honest placeholder.
  Zero-hallucination (CLAUDE.md §10): keys are pointers into ``paper.bib`` — the
  single source of truth — never a generated citation string. Resolve a key to
  full BibTeX with ``sp.bibtex(keys=[...])``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, Dict, Tuple

from .core.results import _to_jsonable


def result_to_dict(obj: Any) -> Dict[str, Any]:
    """Return a JSON-safe ``{field: value}`` mapping for a result dataclass.

    Every field is passed through ``core.results._to_jsonable`` so numpy
    scalars/arrays, pandas Series/DataFrames and NaN/Inf are converted to
    JSON-friendly values (``json.dumps`` never raises on the result).
    """
    if not is_dataclass(obj):
        raise TypeError(
            f"result_to_dict expects a result dataclass, got " f"{type(obj).__name__}."
        )
    return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}


def _fmt_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return r"\text{" + str(v) + "}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def result_to_latex(
    obj: Any, *, caption: str | None = None, label: str | None = None
) -> str:
    """Render the *scalar* fields of a result dataclass as a booktabs table.

    Array / DataFrame / nested fields are summarised as their shape rather than
    dumped, so the table stays publication-sized. Mirrors the ``to_latex``
    other result objects expose, so ``sp.<est>(...).to_latex()`` works
    uniformly.
    """
    d = result_to_dict(obj)
    rows = []
    for key, val in d.items():
        if val is None:
            continue
        if isinstance(val, (list, dict)):
            n = len(val)
            disp = f"[{n} item{'s' if n != 1 else ''}]"
        else:
            disp = _fmt_scalar(val)
        rows.append((str(key).replace("_", r"\_"), disp))
    cap = caption or type(obj).__name__
    body = " \\\\\n".join(f"  {k} & {v}" for k, v in rows)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{cap}}}",
    ]
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Field & Value \\\\",
        "\\midrule",
        body + " \\\\" if body else "",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


class ResultProtocolMixin:
    """Mixin giving a result dataclass ``to_dict`` / ``to_latex`` / ``cite``.

    Set ``_citation_keys`` to the paper.bib key(s) of the method's canonical
    reference(s) — verified to exist in ``paper.bib`` (CLAUDE.md §10) — to make
    ``cite()`` return them; leave it empty for textbook methods with no single
    canonical paper.
    """

    #: Verified paper.bib key(s) for the estimator (see CLAUDE.md §10).
    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of every field (agent-native serialization)."""
        return result_to_dict(self)

    def to_latex(self, *, caption: str | None = None, label: str | None = None) -> str:
        """A compact booktabs table of the result's scalar fields."""
        return result_to_latex(self, caption=caption, label=label)

    def to_word(self, filename: str, title: str | None = None) -> None:
        """§3 Word (.docx) export (alias of ``to_docx``).

        Fulfils the StatsPAI §3 result contract — every result class exposes
        ``to_word`` regardless of which specific mixin/parent supplies the
        underlying implementation. Delegates to the class's own ``to_docx`` if
        it defines one; otherwise raises a clear ``NotImplementedError`` so the
        contract audit (see ``scripts/result_protocol_audit.py``) can flag the
        gap explicitly.
        """
        fn = getattr(self, "to_docx", None)
        if fn is None or getattr(fn, "__isabstractmethod__", False):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement to_docx; override "
                "to_word or add a to_docx method to satisfy the §3 contract."
            )
        return fn(filename, title)

    def to_excel(self, path: str, **kwargs: Any) -> Any:
        """§3 Excel export (alias of ``to_excel``).

        Same delegation pattern as ``to_word``: re-export the host class's
        own ``to_excel`` so the §3 method name is universally available.
        """
        fn = getattr(self, "to_excel", None)
        if fn is None or getattr(fn, "__isabstractmethod__", False):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement to_excel; override "
                "to_excel or add one to satisfy the §3 contract."
            )
        return fn(path, **kwargs)

    def cite(self, format: str = "keys") -> Any:
        """Return the estimator's verified paper.bib citation key(s).

        Parameters
        ----------
        format : {"keys", "json"}, default ``"keys"``
            ``"keys"`` returns a newline-joined string of bib keys (or an
            honest placeholder); ``"json"`` returns a structured dict.

        Notes
        -----
        Zero-hallucination (CLAUDE.md §10): the keys point into ``paper.bib``;
        resolve them to full BibTeX via ``sp.bibtex(keys=[...])``. A method
        with no single canonical paper honestly returns a placeholder rather
        than a fabricated reference.
        """
        keys = tuple(getattr(self, "_citation_keys", ()) or ())
        if format == "json":
            return {
                "citation_keys": list(keys),
                "source": "paper.bib",
                "resolve_with": "sp.bibtex(keys=[...])",
            }
        if format != "keys":
            raise ValueError(f"format must be 'keys' or 'json'; got {format!r}")
        if not keys:
            return (
                f"No single canonical reference registered for "
                f"{type(self).__name__}; see the estimator docstring."
            )
        return "\n".join(keys)
