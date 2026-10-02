"""Python replay strings for agent tool calls.

Every estimator reached through the agent layer (curated tools and the
auto-registered registry tools) reports ``replay``: the ``sp.<fn>(...)``
call that reproduces the fit in a Python session, built from the
arguments that actually reached the estimator (arguments dropped as
unsupported are not in it). The dataset is written as ``data=data``; the
MCP layer appends a comment naming where ``data`` came from (the
``data_path``, ``data_id`` or inline table), so a human can re-run the
analysis outside the agent loop.
"""

from __future__ import annotations

import keyword
import re
from typing import Any, Dict, Mapping, Optional


def _literal(value: Any) -> str:
    """Python literal for a JSON-like value; a placeholder otherwise."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_literal(v) for v in value)
        if isinstance(value, tuple):
            return f"({inner}{',' if len(value) == 1 else ''})"
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return f"<{type(value).__name__}>"


def build_replay(
    fn_name: str,
    kwargs: Mapping[str, Any],
    *,
    result_id: Optional[str] = None,
) -> str:
    """Return ``sp.<fn_name>(data=data, key=value, ...)``.

    Parameters
    ----------
    fn_name : str
        Public ``statspai`` function name.
    kwargs : mapping
        Keyword arguments as passed to the function. ``data`` (any value)
        renders as ``data=data``; a fitted ``result`` injected from a
        handle renders as ``result=result_<result_id>``; keys starting
        with ``_`` (server-internal) are skipped.
    result_id : str, optional
        Handle the ``result`` argument came from.
    """
    parts = []
    if "data" in kwargs:
        parts.append("data=data")
    for key, value in kwargs.items():
        if key == "data" or str(key).startswith("_"):
            continue
        if not str(key).isidentifier() or keyword.iskeyword(str(key)):
            continue
        if key == "result" and result_id:
            parts.append(f"result=result_{result_id}")
            continue
        parts.append(f"{key}={_literal(value)}")
    return f"sp.{fn_name}({', '.join(parts)})"


def data_comment(
    *,
    data_path: Optional[str] = None,
    data_id: Optional[str] = None,
    inline_provenance: Optional[Dict[str, Any]] = None,
    data_columns: Optional[Any] = None,
    data_sample_n: Optional[Any] = None,
) -> str:
    """Comment naming the source of ``data`` in a replay string."""
    if data_id:
        src = f"data_id {data_id!r} (lineage in data_provenance)"
    elif data_path:
        src = f"data_path {data_path!r}"
    elif inline_provenance:
        sha = str(inline_provenance.get("sha256", ""))[:12]
        src = f"inline {inline_provenance.get('format', 'table')} (sha256 {sha}...)"
    else:
        return ""
    extras = []
    if data_columns:
        extras.append(f"columns={list(data_columns)!r}")
    if data_sample_n is not None:
        extras.append(f"sample_n={data_sample_n!r}, seed=0")
    tail = f" [{', '.join(extras)}]" if extras else ""
    return f"  # data = {src}{tail}"


#: ``replay_completeness.level`` values, weakest first.
REPLAY_LEVELS = ("call_only", "session_replayable", "standalone")

_PLACEHOLDER = re.compile(r"=<[A-Za-z_][A-Za-z0-9_]*>")


def replay_completeness(
    replay: str,
    data_provenance: Optional[Mapping[str, Any]] = None,
    *,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """How far a ``replay`` string goes towards reproducing the call.

    Returns ``{"level", "needs"}``. ``level`` is

    * ``"standalone"`` -- a new Python process can re-run the call from
      the string, given what ``needs`` lists (a local file, identified by
      path and SHA-256, or nothing for a call that takes no data);
    * ``"session_replayable"`` -- it depends on a handle held by this
      server process (a ``data_id`` whose lineage is in
      ``data_provenance``, or a fitted ``result_id``);
    * ``"call_only"`` -- the string documents the call but cannot re-run
      it: an argument had no literal form, or the data was sent inline
      or fetched from a URL and only a hash (or nothing) was kept.

    The weakest component decides the level.
    """
    rank = len(REPLAY_LEVELS) - 1
    needs = []

    def _cap(level: str, need: str) -> None:
        nonlocal rank
        rank = min(rank, REPLAY_LEVELS.index(level))
        needs.append(need)

    if _PLACEHOLDER.search(replay):
        _cap("call_only", "argument values shown as <TypeName> have no literal form")
    if result_id and f"result_{result_id}" in replay:
        _cap("session_replayable", f"fitted result {result_id} from this session")
    if "data=data" in replay:
        prov = data_provenance or {}
        kind = prov.get("source_type")
        if kind == "handle":
            _cap(
                "session_replayable",
                f"dataset handle {prov.get('data_id')}; re-apply "
                "data_provenance.lineage to rebuild it in a new process",
            )
        elif kind == "local":
            sha = prov.get("sha256")
            needs.append(
                f"file {prov.get('source')}"
                + (f" (sha256 {sha})" if sha else " (not hashed)")
            )
        elif kind == "inline":
            _cap("call_only", "inline data: only its hash was recorded")
        elif kind == "remote":
            _cap("call_only", "remote data is not hashed; its content may change")
        else:
            _cap("call_only", "the source of `data` was not recorded")
    return {"level": REPLAY_LEVELS[rank], "needs": needs}


# --------------------------------------------------------------------------- #
#  Reproduction bundle
# --------------------------------------------------------------------------- #
def replay_transforms(data: Any, steps: Any) -> Any:
    """Re-apply recorded ``transform_data`` steps to a DataFrame, in order.

    ``steps`` is the ``steps`` list of a reproduction bundle: each item is
    ``{"op": ..., **arguments}`` exactly as the server ran it.
    """
    from .workflow_tools import _apply_transform

    for step in steps:
        data = _apply_transform(data, dict(step))
    return data


def _headline(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The numbers a re-run is checked against."""
    expected: Dict[str, Any] = {}
    for key in ("estimate", "std_error", "n_obs"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            expected[key] = value
    coefs = payload.get("coefficients")
    if isinstance(coefs, Mapping):
        table = {
            str(name): row.get("estimate")
            for name, row in coefs.items()
            if isinstance(row, Mapping)
            and isinstance(row.get("estimate"), (int, float))
        }
        if table:
            expected["coefficients"] = table
    return expected


_SCRIPT = """\
# Reproduction bundle for StatsPAI result {result_id}
# written by statspai {version}. Run in a fresh Python process:
#     python this_file.py
# It checks the data file against the recorded SHA-256, rebuilds the
# analysis data, re-runs the call and compares the headline numbers.
import hashlib
import math

import statspai as sp
from statspai.agent._data_loader import load_dataframe
from statspai.agent._replay import replay_transforms

SOURCE = {source!r}
SHA256 = {sha256!r}
STEPS = {steps!r}
EXPECTED = {expected!r}

with open(SOURCE, "rb") as fh:
    digest = hashlib.sha256(fh.read()).hexdigest()
if digest != SHA256:
    raise SystemExit(f"{{SOURCE}} is not the file that was analysed (sha256 {{digest}})")

data = load_dataframe(SOURCE, columns={columns!r}, sample_n={sample_n!r})
data = replay_transforms(data, STEPS)
result = {call}


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


checked = 0
if "estimate" in EXPECTED and hasattr(result, "estimate"):
    assert _close(float(result.estimate), EXPECTED["estimate"]), "estimate differs"
    checked += 1
for name, value in EXPECTED.get("coefficients", {{}}).items():
    assert _close(float(result.params[name]), value), f"coefficient {{name}} differs"
    checked += 1
print(f"reproduced: {{checked}} number(s) match")
"""


def build_bundle(
    *,
    result_id: str,
    replay: str,
    data_provenance: Optional[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Everything a new process needs to re-run a cached result, or why not.

    Returns ``{"completeness", "needs", "source", "steps", "call",
    "expected", "script"}``. ``script`` is present only when the bundle is
    ``standalone``: the data came from a local file whose hash was
    recorded (directly, or through a chain of ``transform_data`` steps),
    and every argument of the call has a literal form. Otherwise
    ``completeness`` and ``needs`` say what is missing, and no script is
    offered that could not run.
    """
    import statspai

    call = replay.split("  #", 1)[0].strip()
    prov = dict(data_provenance or {})
    needs = []
    root: Dict[str, Any] = {}
    steps: list = []
    if prov.get("source_type") == "handle":
        root = dict(prov.get("root") or {})
        for item in reversed(prov.get("lineage") or []):
            if item.get("missing"):
                needs.append(f"dataset handle {item.get('data_id')} was evicted")
            for op in item.get("operations") or []:
                steps.append({"op": op.get("op"), **dict(op.get("arguments") or {})})
    elif prov.get("source_type") == "local":
        root = prov
    elif "data=data" in call:
        needs.append(
            "the data was sent inline or fetched remotely; only a hash "
            "(or nothing) was recorded"
            if prov
            else "the source of `data` was not recorded"
        )
    if "data=data" in call and not needs:
        if root.get("source_type") != "local":
            needs.append("the data does not trace back to a local file")
        elif not root.get("sha256"):
            needs.append("the data file was not hashed")
    if _PLACEHOLDER.search(call):
        needs.append("argument values shown as <TypeName> have no literal form")
    if "result=result_" in call:
        needs.append("the call consumes another fitted result held by the server")
    bundle: Dict[str, Any] = {
        "bundle_version": 1,
        "result_id": result_id,
        "statspai_version": statspai.__version__,
        "completeness": "standalone" if not needs else "call_only",
        "needs": needs,
        "source": {
            k: root[k]
            for k in ("source", "sha256", "format", "columns_requested", "sample_n")
            if k in root
        },
        "steps": steps,
        "call": call,
        "expected": _headline(payload),
    }
    if not needs and "data=data" in call:
        bundle["script"] = _SCRIPT.format(
            result_id=result_id,
            version=statspai.__version__,
            source=root.get("source"),
            sha256=root.get("sha256"),
            steps=steps,
            expected=bundle["expected"],
            columns=root.get("columns_requested"),
            sample_n=root.get("sample_n"),
            call=call,
        )
    return bundle


__all__ = [
    "REPLAY_LEVELS",
    "build_bundle",
    "build_replay",
    "data_comment",
    "replay_completeness",
    "replay_transforms",
]
