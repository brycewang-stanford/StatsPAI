"""Guard: documented ``sp.*(...)`` calls must match the real signatures.

Two classes of rot this catches, both of which shipped undetected:

* ``choosing_did_estimator.md`` advertised
  ``sp.stacked_did(df, y, g, t, i, event_window=6)``.  There is no
  ``event_window`` parameter -- the name was copied from the neighbouring
  ``sp.sun_abraham`` row -- so the documented call raised ``TypeError``.
* ``migration-from-r.md`` mapped ``HonestDiD::createSensitivityResults`` to
  ``sp.honest_did(cs_result, Mbar=...)``.  ``Mbar`` belongs to
  ``sp.sensitivity_rr`` on the next row; ``honest_did`` takes ``m_grid``.

What is *deliberately* not checked
----------------------------------
Guides elide arguments as prose -- ``sp.dml(...)``, ``sp.paper()``,
``sp.callaway_santanna(..., estimator='dr')``.  A "missing required
argument" is therefore never an error here, and a bare ``...`` positional
is treated as an elision marker rather than a real argument.

Family dispatchers (``sp.decompose(method=...)``, ``sp.mr(method=...)``)
declare their selector positional-only so ``**kwargs`` can carry a
same-named argument through to the target estimator.  The guides still
write them in ``method=`` notation, which is the documented house style,
so a keyword naming a real-but-positional-only parameter passes.

Scope
-----
Both ``docs/guides/`` and the published ``docs/reference/`` pages (plus
``docs/index.md``) are held to zero tolerance.  The reference pages had
87 wrong-keyword snippets written against an older API -- ``sp.sar(df,
y=, x=, W=)`` when the signature is ``sar(W, data, formula)``,
``sp.cox(time=)`` when it is ``duration=`` -- and a handful that
documented parameters never implemented at all (``sp.cox``'s
counting-process ``time_start``/``time_stop``, ``sp.garch(model=)``).
Those were fixed or removed; the count is now zero and stays there.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import warnings

import pytest

import statspai as sp

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GUIDES = _ROOT / "docs" / "guides"
_REFERENCE_PAGES = sorted((_ROOT / "docs" / "reference").glob("*.md")) + [
    _ROOT / "docs" / "index.md"
]

# Inline code spans of the form `sp.name(...)`.
_SPAN = re.compile(r"`(sp\.[A-Za-z_][A-Za-z0-9_]*\([^`]*\))`")

# Fenced blocks (```python, ```py, or bare ```), and any sp.<name> token.
_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.S)
_TOKEN = re.compile(r"\bsp\.([a-zA-Z_][a-zA-Z0-9_]*)")

# ``sp.<name>`` tokens that are prose, not API references.
_NOT_API_REFERENCES = {
    # Documented in rigorous_lasso_hdm.md's "Not yet ported" section as the
    # name StatsPAI deliberately does NOT expose; the IIVM score in sp.dml
    # covers it instead.  The whole point of the sentence is its absence.
    "rlasso_latet",
    # survey_ph.md writes the glob `sp.svy*` to mean the svyglm / svymean /
    # svytotal family, each of which does resolve.
    "svy",
    # replication_workflow.md template placeholder for the reader's own call.
    "your_estimator",
    # docs/index.md stand-in for "whatever estimator you picked".
    "someestimator",
}

# Calls a guide shows *in order to say they are wrong*.  Keyed by
# (guide filename, exact source text) so a genuine future typo elsewhere
# still fails.
_INTENTIONALLY_INVALID = {
    (
        "mixtape_replications.md",
        "sp.regtable(..., column_labels=...)",
    ),
}


def _parse(src, mode="exec"):
    try:
        with warnings.catch_warnings():
            # Snippets carry table-escaped text such as `\|`, which the
            # compiler flags as an invalid escape sequence.
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(src, mode=mode)
    except SyntaxError:
        # Prose such as `sp.foo(y ~ x)`, or a shell/output block.
        return None


def _documented_calls():
    """Yield ``(guide_name, source, ast.Call)`` for every ``sp.*`` call in an
    inline code span *or* a fenced code block, across both surfaces."""
    for md in sorted(_GUIDES.glob("*.md")) + _REFERENCE_PAGES:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")

        for src in _SPAN.findall(text):
            tree = _parse(src, mode="eval")
            if tree is None:
                continue
            node = tree.body
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                yield md.name, src, node

        for block in _FENCE.findall(text):
            tree = _parse(block)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Only `sp.<name>(...)`, not `sp.iv.<name>(...)` or `r.plot()`.
                if not (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "sp"
                ):
                    continue
                yield md.name, f"sp.{func.attr}(...)", node


def _resolve(node: ast.Call):
    """Return the signature of the called ``sp.<name>``, or None."""
    fn = getattr(sp, node.func.attr, None)
    if fn is None or not callable(fn):
        return None
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C extensions
        return None


def test_every_documented_sp_name_resolves():
    """A guide must not tell the reader to call something that is not there.

    This caught eight dead recommendations at once -- ``sp.weak_iv_ci``,
    ``sp.post_lasso``, ``sp.jive_variants``, ``sp.plausibly_exogenous``,
    ``sp.mte``, ``sp.ivmte_lp`` (all real code, but reachable only as
    ``sp.iv.<name>``), plus ``sp.rdmulti``, ``sp.pcalg``, ``sp.S_Learner``,
    ``sp.bootstrap_ci``, ``sp.permutation_test``, ``sp.audit_result``,
    ``sp.vecm`` and a doubled ``sp.sp.`` prefix.
    """
    bad = []
    for md in sorted(_GUIDES.glob("*.md")) + _REFERENCE_PAGES:
        if not md.exists():
            continue
        for name in sorted(set(_TOKEN.findall(md.read_text(encoding="utf-8")))):
            if name in _NOT_API_REFERENCES:
                continue
            if getattr(sp, name, None) is None:
                bad.append(f"{md.name}: sp.{name} does not resolve")
    assert not bad, "doc references a nonexistent symbol:\n  " + "\n  ".join(bad)


def test_guides_expose_some_calls_to_check():
    """Fail loudly if the regex stops matching -- an empty sweep passes
    every other assertion vacuously."""
    calls = list(_documented_calls())
    assert len(calls) > 100, f"only {len(calls)} inline sp.* calls found"


def test_documented_keywords_are_real_parameters():
    bad = []
    for guide, src, node in _documented_calls():
        if (guide, src) in _INTENTIONALLY_INVALID:
            continue
        sig = _resolve(node)
        if sig is None:
            continue
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue  # **kwargs absorbs anything
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in params:
                bad.append(f"{guide}: {src}\n    -> no parameter {kw.arg!r}")
    assert not bad, "documented keyword does not exist:\n  " + "\n  ".join(bad)


def test_documented_positional_counts_fit_the_signature():
    bad = []
    for guide, src, node in _documented_calls():
        if (guide, src) in _INTENTIONALLY_INVALID:
            continue
        sig = _resolve(node)
        if sig is None:
            continue
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
            continue  # *args absorbs anything
        slots = sum(
            p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for p in params.values()
        )
        # A bare `...` is an elision marker in prose, not an argument.
        given = sum(
            not isinstance(a, ast.Constant) or a.value is not ... for a in node.args
        )
        if given > slots:
            bad.append(f"{guide}: {src}\n    -> {given} positional args, {slots} slots")
    assert not bad, "too many positional arguments:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize(
    "guide, src",
    sorted(_INTENTIONALLY_INVALID),
)
def test_intentionally_invalid_examples_still_present(guide, src):
    """If a guide stops showing a deliberate counter-example, drop it from
    the allowlist rather than letting the entry silently rot."""
    text = (_GUIDES / guide).read_text(encoding="utf-8")
    assert src in text, f"{guide} no longer contains {src!r}; prune the allowlist"
