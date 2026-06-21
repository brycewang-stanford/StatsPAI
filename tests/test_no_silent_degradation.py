"""Lint guards against CLAUDE.md §3.7 violations: orchestration code must
not silently swallow exceptions, and tests must not silently skip on
"runtime path differs" exceptions.

The repo has accumulated a small amount of historical debt that fits these
patterns. The constants below freeze today's count; the assertions then
ratchet — the count cannot grow, but it is allowed (and expected) to shrink
as the debt is paid down. Updating the baseline downward in the same PR that
fixes a violation is the intended workflow.

Patterns checked:

1. ``except Exception: pass`` (bare swallow) anywhere under
   ``src/statspai/workflow/``, ``src/statspai/smart/``,
   ``src/statspai/agent/`` — orchestration paths that §3.7 explicitly
   prohibits from silent degradation.

2. ``except Exception: <name> = None`` (silent None fallback) in the same
   paths.

3. ``pytest.skip("...path differs..." | "...runtime: ...")`` in test files —
   the dead-defensive scaffold that hides estimator regressions instead of
   surfacing them.

Each helper returns the list of (file, lineno, snippet) for transparency;
updating the baselines is intentional and visible in the test diff.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# CLAUDE.md §3.7 explicitly scopes the "must call record_degradation" rule to
# orchestration / best-effort paths: ``workflow/``, ``smart/``, and ``paper``.
# ``agent/`` is a system boundary (JSON-RPC / MCP translation) where catching
# Exception and translating to an error response is the correct behaviour, so
# the lint deliberately does not scan it.
ORCHESTRATION_ROOTS = [
    REPO_ROOT / "src" / "statspai" / "workflow",
    REPO_ROOT / "src" / "statspai" / "smart",
]
TEST_ROOTS = [REPO_ROOT / "tests"]

# Frozen as of 2026-05-28. Decrement (in the same PR that removes the
# violation) when paid down. NEVER increment without an explicit comment
# explaining why a new silent-degradation path is acceptable.
#
# Current debt snapshot:
#   bare swallow (4):
#     smart/session.py:153, 170, 176, 198
#     (smart/citations.py:422 paid down 2026-06-21 — unsupported
#      cite(format="json") fallbacks now catch specific exceptions)
#     (smart/verify.py:118, 148 paid down 2026-06-21 — array coercion
#      fallbacks now catch TypeError/ValueError only)
#     (workflow/causal_workflow.py:667 paid down 2026-06-10 —
#      _extract_effect now routes through record_degradation)
#   silent None fallback (4):
#     workflow/causal_workflow.py:662, 886
#     workflow/paper.py:1043, 1255
BARE_SWALLOW_MAX = 4
SILENT_NONE_MAX = 4
PATH_DIFFERS_SKIP_MAX = 0  # cleaned up 2026-05-28; do not let it regrow


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from sorted(root.rglob("*.py"))


def _handler_targets_exception(handler: ast.ExceptHandler) -> bool:
    """True if the handler matches ``Exception`` or ``BaseException``."""
    t = handler.type
    if t is None:  # bare ``except:``
        return True
    if isinstance(t, ast.Name):
        return t.id in {"Exception", "BaseException"}
    if isinstance(t, ast.Attribute):
        return t.attr in {"Exception", "BaseException"}
    return False


def _handler_body_just_passes(handler: ast.ExceptHandler) -> bool:
    return len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)


def _handler_body_just_assigns_none(handler: ast.ExceptHandler) -> bool:
    """True if the handler body is a single ``X = None`` assignment."""
    if len(handler.body) != 1:
        return False
    stmt = handler.body[0]
    if not isinstance(stmt, ast.Assign):
        return False
    val = stmt.value
    if isinstance(val, ast.Constant) and val.value is None:
        return True
    return False


def _scan_orchestration() -> tuple[list[str], list[str]]:
    bare, none_assign = [], []
    for path in _iter_python_files(ORCHESTRATION_ROOTS):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not _handler_targets_exception(node):
                continue
            rel = path.relative_to(REPO_ROOT)
            tag = f"{rel}:{node.lineno}"
            if _handler_body_just_passes(node):
                bare.append(tag)
            elif _handler_body_just_assigns_none(node):
                none_assign.append(tag)
    return bare, none_assign


_SKIP_PATTERN = re.compile(
    r"pytest\.skip\([^)]*(?:path differs|runtime:)[^)]*\)",
    re.IGNORECASE,
)


_SELF_PATH = Path(__file__).resolve()


def _scan_path_differs_skips() -> list[str]:
    hits = []
    for path in _iter_python_files(TEST_ROOTS):
        if path.resolve() == _SELF_PATH:
            # This file's docstring / error messages mention the very
            # phrase the regex matches; exclude self to avoid false hits.
            continue
        text = path.read_text(encoding="utf-8")
        for m in _SKIP_PATTERN.finditer(text):
            # Recover line number
            line_no = text.count("\n", 0, m.start()) + 1
            rel = path.relative_to(REPO_ROOT)
            hits.append(f"{rel}:{line_no}")
    return hits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scan():
    bare, none_assign = _scan_orchestration()
    return {
        "bare": bare,
        "none_assign": none_assign,
        "path_differs": _scan_path_differs_skips(),
    }


def test_bare_exception_swallow_does_not_regrow(scan):
    """``except Exception: pass`` in orchestration paths must not grow.

    Violates §3.7: orchestration must call ``record_degradation`` so the
    user / agent sees what failed. ``pass`` is exactly the silent-failure
    pattern the rule was written to forbid.
    """
    hits = scan["bare"]
    assert len(hits) <= BARE_SWALLOW_MAX, (
        f"Bare ``except Exception: pass`` count grew from "
        f"{BARE_SWALLOW_MAX} to {len(hits)}.\n"
        "Either fix the new violation by routing the failure through "
        "``record_degradation``, or — if intentional — decrement "
        "BARE_SWALLOW_MAX is the wrong direction here; this counter only "
        "moves DOWN when debt is paid.\nViolations:\n  " + "\n  ".join(sorted(hits))
    )


def test_silent_none_fallback_does_not_regrow(scan):
    """``except Exception: <name> = None`` in orchestration must not grow.

    Like bare swallow, this hides degradation. The fix is to also call
    ``record_degradation`` so downstream consumers can surface the issue
    in ``.degradations`` and the ``WorkflowDegradedWarning`` stream.
    """
    hits = scan["none_assign"]
    assert len(hits) <= SILENT_NONE_MAX, (
        f"Silent ``except Exception: X = None`` count grew from "
        f"{SILENT_NONE_MAX} to {len(hits)}.\n"
        "Wrap with ``record_degradation(...)`` before the assignment.\n"
        "Violations:\n  " + "\n  ".join(sorted(hits))
    )


def test_no_path_differs_skip_pattern_regrowth(scan):
    """Test files must not regrow ``pytest.skip("...path differs...")``.

    This dead-defensive scaffold hides estimator regressions: an estimator
    that quietly raises ``KeyError`` is reported as "skipped" rather than
    "failed", so the regression slips into a release. Cleaned up in
    2026-05-28; floor stays at zero.
    """
    hits = scan["path_differs"]
    assert len(hits) <= PATH_DIFFERS_SKIP_MAX, (
        f"Found {len(hits)} ``pytest.skip('...path differs...')`` call(s); "
        f"baseline is {PATH_DIFFERS_SKIP_MAX}.\n"
        "If the test legitimately cannot run, mark it ``@pytest.mark.xfail("
        "strict=True, reason=...)`` or assert the expected exception.\n"
        "Violations:\n  " + "\n  ".join(sorted(hits))
    )
