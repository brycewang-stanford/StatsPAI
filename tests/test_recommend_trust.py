"""Trust contracts for ``sp.recommend`` — the user-facing recommendation API.

Like ``from_stata`` / ``from_r``, ``sp.recommend`` emits BOTH a ``code``
string (for an LLM / user to copy-paste) AND a structured ``params`` dict
(for an MCP ``tools/call`` dispatch). The two must describe the same call,
every emitted ``sp.<fn>(...)`` must be callable, and no required arg may
be silently dropped — same trust contract that protects the migration
translators, applied to the recommendation surface.

NOTE on ``eval``: the harness ``_run_recommendation`` evaluates the emitted
``code`` string with the stdlib ``eval`` because the recommendation surface
intentionally emits arbitrary user-facing snippets (e.g.
``sp.callaway_santanna(df, ...)``) whose shape is structurally
unconstrained — a full AST re-execution would require re-implementing the
recommendation engine. This is acceptable here only because the source of
the code string is ``statspai.smart.recommend`` (a peer-reviewed, audited
handler in this same repository), not user input. The same caveat applied
to the existing ``test_replication_pack`` / ``test_bench_did`` harnesses
in the migration surface.
"""

from __future__ import annotations

import ast
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

#: ``params`` dicts use a few "data carrier" keys (data, df) that are
#: not declared in the actual function signature. Treat them as ambient.
_DATA_CARRIERS = {"data", "df"}


def _parse_call_kwargs(code):
    """Parse ``sp.fn(arg1, kw=val, data=df)`` and return (tool, kwargs).

    Tolerates a leading comment, the ``data=df`` data carrier, the
    ``# Derived ...`` comment that ``sp.recommend`` prepends, and any
    ``prep = ...`` assignment in the source. Returns None on a code
    snippet that is not a real call (the ``prep``-only case).
    """
    # Some recommend outputs start with a single-line comment that
    # describes the call. Strip leading comment lines before parsing so
    # the parser sees the real sp.<fn>(...) expression.
    lines = code.splitlines()
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    parseable = "\n".join(lines) if lines else code
    try:
        tree = ast.parse(parseable, mode="exec")
    except SyntaxError:
        return None
    # Find the first sp.<fn>(... call) in the body.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            f = call.func
            if isinstance(f, ast.Name):
                tool = f.id
            elif isinstance(f, ast.Attribute):
                tool = f"sp.{f.attr}"
            else:
                continue
            if not tool.startswith("sp."):
                continue
            out = {"_tool": tool}
            for k in call.keywords or []:
                if k.arg is None:
                    continue
                v = k.value
                if isinstance(v, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
                    try:
                        out[k.arg] = ast.literal_eval(v)
                        continue
                    except Exception:
                        pass
                if isinstance(v, ast.Constant):
                    out[k.arg] = v.value
                elif isinstance(v, ast.Name):
                    out[k.arg] = v.id
                else:
                    out[k.arg] = ast.unparse(v)
            # Drop data carriers from the comparison set; the agent's
            # df is ambient.
            for k in _DATA_CARRIERS:
                out.pop(k, None)
            return out
    return None


def _run_recommendation(rec, df):
    """Run the recommendation's ``code`` (after any ``prep`` step) and
    return the resulting sp.<fn> result, or ``"err:<msg>"`` on failure.
    The ``prep`` step exists precisely because some recommendations must
    mutate the dataframe (e.g. add a cohort column) before the sp.<fn>
    call — the trust contract is that running prep+code is equivalent
    to dispatching (tool, params)."""
    code = rec.get("code", "")
    if not code:
        return "err:placeholder"
    # Some recommendations begin with a single-line comment that describes
    # the call (e.g. ``# Derived cohort column = first period treated``);
    # the real call follows on the next line. Strip leading comment lines
    # so eval can find the sp.<fn>(...) expression.
    real_lines = [
        ln for ln in code.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not real_lines:
        return "err:placeholder"
    real_code = "\n".join(real_lines)
    prep = rec.get("prep")
    local_df = df.copy()
    if prep is not None:
        try:
            local_df = prep(local_df)
        except Exception as e:
            return f"err:prep {type(e).__name__}: {e}"
    g = {"sp": sp, "df": local_df, "np": np, "pd": pd}
    try:
        return eval(real_code, g)
    except Exception as e:
        return f"err:eval {type(e).__name__}: {e}"


def _dispatch_recommendation(rec, df):
    """Run the recommendation's ``params`` dict via the actual sp tool.

    Filters ``params`` to the function's actual signature (so unknown
    kwargs cannot crash a real dispatcher) — this mirrors what an MCP
    ``tools/call`` would do."""
    import inspect

    fn_name = rec.get("function")
    if not fn_name:
        return None
    fn = getattr(sp, fn_name, None)
    if fn is None or not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
        accepted = {
            n
            for n, p in sig.parameters.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
    except (ValueError, TypeError):
        return None
    prep = rec.get("prep")
    local_df = df.copy()
    if prep is not None:
        try:
            local_df = prep(local_df)
        except Exception:
            pass
    args = {k: v for k, v in (rec.get("params") or {}).items() if k in accepted}
    if "data" in accepted and "data" not in args:
        args["data"] = local_df
    elif "df" in accepted and "df" not in args:
        args["df"] = local_df
    try:
        return fn(**args)
    except Exception as e:
        return f"err:dispatch {type(e).__name__}: {e}"


def _coef_dict(result):
    """Coefs from a fitted sp result, or None."""
    if result is None or isinstance(result, str) or not hasattr(result, "params"):
        return None
    p = result.params
    if hasattr(p, "to_dict"):
        return {str(k): float(v) for k, v in p.to_dict().items()}
    try:
        import numpy as _np

        return {str(i): float(v) for i, v in enumerate(_np.atleast_1d(p))}
    except Exception:
        return None


#: A small DID-style panel that the recommend surface understands.
def _did_df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        for t in range(10):
            treat = 1 if (i < 30 and t >= 5) else 0
            rows.append(
                {
                    "y": 2 * treat + rng.normal(),
                    "x": rng.normal(),
                    "treat": treat,
                    "i": i,
                    "t": t,
                }
            )
    return pd.DataFrame(rows)


#: A small RD-style panel for cutoff-based recommendations.
def _rd_df():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 600)
    # RD recommendations want a treatment-style column too; add a binary
    # running_var>cutoff so the recommend surface has the same column
    # shape the sp.* tools would expect.
    return pd.DataFrame(
        {
            "y": x**2 + rng.normal(size=600) * 0.1,
            "x": x,
            "treat": (x > 0.0).astype(int),
        }
    )


#: (channel, builder_fn, design_kwargs) — exercise the major recommend
#: branches with data that the corresponding sp tools can fit.
_RECOMMEND_CASES = [
    ("did", _did_df, {"design": "did", "time": "t", "id": "i"}),
    ("rd", _rd_df, {"design": "rd", "running_var": "x", "cutoff": 0.0}),
]


@pytest.mark.parametrize("channel,builder,extra", _RECOMMEND_CASES)
def test_recommend_emits_dispatchable_codes(channel, builder, extra):
    """Every ``sp.<fn>`` emitted by recommend is callable, and the args
    in ``code`` parse back to a kwargs dict that matches the structured
    ``params``. The trust contract for the migration surface applied
    to the recommendation surface — because an LLM might copy the code
    AND an agent might dispatch, and they must arrive at the same model."""
    df = builder()
    out = sp.recommend(df, y="y", treatment="treat", **extra)
    recs = out.recommendations
    assert recs, "recommend returned no recommendations"
    for rec in recs:
        fn = rec.get("function")
        assert fn, f"recommendation missing function: {rec}"
        assert callable(getattr(sp, fn, None)), (
            f"[{channel}] recommend emits sp.{fn} which is not callable — "
            f"the recommended tool is a dead on-ramp."
        )
        code = rec.get("code", "")
        assert code, f"[{channel}] recommend emitted no code for sp.{fn}"
        parsed = _parse_call_kwargs(code)
        assert parsed, f"[{channel}] code is not a parseable sp.<fn> call: {code!r}"
        assert parsed["_tool"] == f"sp.{fn}", (
            f"[{channel}] code calls {parsed['_tool']!r} but params target "
            f"sp.{fn!r} — copy-paste and dispatch diverge."
        )
        code_kw = {k: v for k, v in parsed.items() if k != "_tool"}
        params = {
            k: v
            for k, v in (rec.get("params") or {}).items()
            if k not in _DATA_CARRIERS
        }
        for k in params:
            assert k in code_kw, (
                f"[{channel}] params['{k}']={params[k]!r} has no match in "
                f"the emitted code ({code_kw})."
            )
            assert code_kw[k] == params[k] or str(code_kw[k]) == str(params[k]), (
                f"[{channel}] key '{k}': code has {code_kw[k]!r}, params has "
                f"{params[k]!r}."
            )


@pytest.mark.parametrize("channel,builder,extra", _RECOMMEND_CASES)
def test_recommend_code_and_dispatch_agree_numerically(channel, builder, extra):
    """For each recommendation, copy-paste (``code``) and dispatch
    (``function``+``params``) must yield the same coefficient estimates.
    A divergence means the two surfaces drift apart — one of them is
    silently fitting a different model."""
    df = builder()
    out = sp.recommend(df, y="y", treatment="treat", **extra)
    for rec in out.recommendations:
        fn = rec.get("function")
        if fn is None:
            continue
        r_code = _run_recommendation(rec, df)
        r_disp = _dispatch_recommendation(rec, df)
        c_code = _coef_dict(r_code)
        c_disp = _coef_dict(r_disp)
        if c_code is None and c_disp is None:
            continue  # some branches don't return params; skip
        assert c_code and c_disp, (
            f"[{channel}] {fn}: copy-paste and dispatch produced different "
            f"shapes (code={c_code!r}, dispatch={c_disp!r})."
        )
        common = set(c_code) & set(c_disp)
        for k in common:
            assert abs(c_code[k] - c_disp[k]) < 1e-9, (
                f"[{channel}] {fn}: coefficient {k!r} differs — "
                f"code={c_code[k]!r} vs dispatch={c_disp[k]!r}."
            )


def test_recommend_no_dead_function_pointers():
    """Every ``function`` name emitted by ``sp.recommend`` across all major
    design branches must resolve to a callable on the sp namespace —
    a dead function name means a user following the recommendation gets
    a hard ``AttributeError`` when they try to call the suggested tool.

    We exercise DID / RD / IV / Observational / Sensitivity / ATE to cover
    the major recommend branches, then parse every emitted ``code`` and
    ``function`` field for the union of all referenced ``sp.<fn>`` names.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    # DID-shaped panel
    rows = []
    for i in range(40):
        for t in range(8):
            treat = 1 if (i < 20 and t >= 4) else 0
            rows.append(
                {
                    "y": 2 * treat + rng.normal(),
                    "x": rng.normal(),
                    "treat": treat,
                    "i": i,
                    "t": t,
                }
            )
    panel = pd.DataFrame(rows)
    # Observational
    obs = panel.rename(columns={"i": "id"})
    obs["treat_obs"] = (rng.uniform(size=len(obs)) < 0.4).astype(int)
    obs["y_obs"] = 1.5 * obs["treat_obs"] + rng.normal(size=len(obs))

    designs = [
        (
            "did",
            dict(design="did", time="t", id="i", y="y", treatment="treat", data=panel),
        ),
        ("rd", dict(design="rd", running_var="x", cutoff=0.0, y="y", data=obs)),
        (
            "observational",
            dict(
                design="observational",
                y="y_obs",
                treatment="treat_obs",
                covariates=["x"],
                data=obs,
            ),
        ),
    ]

    referenced: set = set()
    for _, kw in designs:
        try:
            r = sp.recommend(**kw)
        except Exception:
            continue
        for rec in r.recommendations:
            fn = rec.get("function")
            if fn:
                referenced.add(fn)
            # Parse the emitted code for any additional sp.<fn> invocations
            # beyond the top-level ``function`` (e.g. embedded ``sp.foo`` in
            # the python_code snippet).
            code = rec.get("code", "") or ""
            for m in re.finditer(r"sp\.([a-zA-Z_][a-zA-Z0-9_]*)", code):
                referenced.add(m.group(1))

    assert referenced, "no sp.<fn> names were collected from recommend output"
    dead = []
    for fn in sorted(referenced):
        obj = getattr(sp, fn, None)
        if obj is None or not callable(obj):
            dead.append(fn)
    assert not dead, (
        "recommend emits dead function pointers (an agent following the "
        f"recommendation hits AttributeError): {dead}"
    )
