"""Runnable code-example surface for StatsPAI agents.

``sp.examples(name)`` is the agent-discoverable entry for "show me how
to call this function". Different from neighbouring APIs:

* :func:`sp.describe_function` returns the full registry record
  (params / assumptions / failure_modes etc.) — useful, but verbose.
* :func:`sp.recommend` walks DATA + research question → estimator
  selection.
* :func:`sp.examples` answers: "I know I want ``sp.{name}``; show me
  one short, copy-pasteable Python snippet that exercises it." —
  agents need this to bootstrap a fresh notebook without reading docs.

Per-method curated snippets cover the flagship surface (regress / did
/ callaway_santanna / rdrobust / ivreg / ebalance / synth /
metalearners). For any other registered function, falls back to the
``example`` field stored on the registry entry.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
#  Curated runnable snippets
# ---------------------------------------------------------------------------
#
# Each entry is keyed by canonical function name and lists 1-2 short,
# self-contained Python blocks. Snippets are formatted as raw triple-
# quoted strings so they round-trip through json.dumps without
# escaping noise.

_CURATED: Dict[str, List[Dict[str, str]]] = {
    "regress": [
        {
            "title": "OLS with HC1 robust standard errors",
            "code": (
                "import numpy as np, pandas as pd\n"
                "import statspai as sp\n\n"
                "rng = np.random.default_rng(0)\n"
                "df = pd.DataFrame({\n"
                "    'wage': rng.normal(loc=10, scale=2, size=300),\n"
                "    'education': rng.integers(8, 20, size=300),\n"
                "    'experience': rng.integers(0, 40, size=300),\n"
                "})\n"
                "df['wage'] += 0.5 * df['education'] "
                "+ 0.1 * df['experience']\n\n"
                "result = sp.regress('wage ~ education + experience', "
                "data=df, robust='hc1')\n"
                "print(result.summary())"
            ),
        },
        {
            "title": "Cluster-robust SEs",
            "code": (
                "result = sp.regress(\n"
                "    'y ~ treat + x', data=df,\n"
                "    cluster='firm_id'  # one-way cluster\n"
                ")"
            ),
        },
    ],
    "did": [
        {
            "title": "Classic 2x2 DID",
            "code": (
                "import numpy as np, pandas as pd\n"
                "import statspai as sp\n\n"
                "rng = np.random.default_rng(0)\n"
                "rows = []\n"
                "for i in range(200):\n"
                "    treated = 1 if i < 100 else 0\n"
                "    for t in (0, 1):  # pre / post\n"
                "        y = (1 + 0.3*t + 0.5*treated\n"
                "             + 2.0*treated*t + rng.normal(scale=0.5))\n"
                "        rows.append({'i': i, 't': t,\n"
                "                     'treated': treated, 'y': y})\n"
                "df = pd.DataFrame(rows)\n\n"
                "r = sp.did(df, y='y', treat='treated', time='t')\n"
                "print(f'ATT = {r.estimate:.3f}, p = {r.pvalue:.3f}')"
            ),
            "expected_output_pattern": "ATT = 2.",
        },
    ],
    "callaway_santanna": [
        {
            "title": "Staggered DID (Callaway-Sant'Anna 2021)",
            "code": (
                "# df has columns: i (unit), t (time), g (cohort: first\n"
                "# treatment period; 0 = never-treated), y (outcome).\n"
                "import statspai as sp\n"
                "r = sp.callaway_santanna(\n"
                "    df, y='y', g='g', t='t', i='i',\n"
                "    estimator='dr',          # doubly robust\n"
                "    control_group='nevertreated',\n"
                ")\n"
                "print(r.summary())"
            ),
        },
    ],
    "rdrobust": [
        {
            "title": "Sharp RD with bias-corrected CIs",
            "code": (
                "import statspai as sp\n"
                "# x is the running variable, c is the cutoff.\n"
                "r = sp.rdrobust(df, y='y', x='score', c=65.0,\n"
                "                kernel='triangular')\n"
                "print(r.summary())"
            ),
        },
    ],
    "ivreg": [
        {
            "title": "Two-stage least squares",
            "code": (
                "import statspai as sp\n"
                "# Wilkinson formula: y ~ exog + (endog ~ instrument)\n"
                "r = sp.ivreg('y ~ x1 + (d ~ z)', data=df,\n"
                "             robust='hc1')\n"
                "print(r.summary())"
            ),
        },
    ],
    "ebalance": [
        {
            "title": "Hainmueller (2012) entropy balancing",
            "code": (
                "import statspai as sp\n"
                "r = sp.ebalance(df, y='y', treat='treated',\n"
                "                covariates=['age', 'education'],\n"
                "                moments=2)  # match means + variances\n"
                "print(r.summary())"
            ),
        },
    ],
    "synth": [
        {
            "title": "Classic synthetic control (Abadie et al. 2010)",
            "code": (
                "import statspai as sp\n"
                "r = sp.synth(df, outcome='y', unit='unit', time='time',\n"
                "             treated_unit='California', treatment_time=1989,\n"
                "             predictors=['cigsale_lag', 'income'])\n"
                "print(r.summary())\n"
                "r.plot('synth')  # treated vs synthetic trajectory"
            ),
        },
    ],
    "audit": [
        {
            "title": "Reviewer checklist of missing robustness evidence",
            "code": (
                "import statspai as sp\n"
                "r = sp.did(df, y='y', treat='treated', time='t')\n"
                "card = sp.audit(r)\n"
                "for c in card['checks']:\n"
                "    if c['status'] == 'missing' and c['importance'] == 'high':\n"
                "        print(f\"  • {c['name']}: try {c['suggest_function']}\")"
            ),
        },
    ],
    "preflight": [
        {
            "title": "Cheap pre-estimation check",
            "code": (
                "import statspai as sp\n"
                "report = sp.preflight(df, 'did',\n"
                "                       y='y', treat='treated', time='t')\n"
                "if report['verdict'] == 'FAIL':\n"
                "    for c in report['checks']:\n"
                "        if c['status'] == 'failed':\n"
                "            print(c['message'])\n"
                "else:\n"
                "    r = sp.did(df, y='y', treat='treated', time='t')"
            ),
        },
    ],
    "detect_design": [
        {
            "title": "Auto-detect study design from a DataFrame",
            "code": (
                "import statspai as sp\n"
                "card = sp.detect_design(df)\n"
                "if card['design'] == 'panel':\n"
                "    unit, time = (card['identified']['unit'],\n"
                "                  card['identified']['time'])"
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def examples(name: str) -> Dict[str, Any]:
    """Return runnable code examples + registry metadata for a function.

    Parameters
    ----------
    name : str
        Canonical StatsPAI function name (e.g. ``"did"``, ``"regress"``,
        ``"callaway_santanna"``). Lower-cased and stripped before lookup.

    Returns
    -------
    dict
        JSON-safe payload with keys:

        - ``name`` (str)
        - ``category`` (str | None) — registry category
        - ``description`` (str) — one-line summary from the registry
        - ``signature`` (str | None) — example call from the registry
        - ``examples`` (list[dict]) — curated runnable snippets (each
          with ``title`` / ``code``); empty if no curated snippet
          exists for this function. The snippets are intentionally
          short (≤ 20 lines) so an agent can paste them whole into a
          fresh REPL.
        - ``pre_conditions`` (list[str])
        - ``assumptions`` (list[str])
        - ``alternatives`` (list[str])
        - ``known_function`` (bool) — ``True`` if registry has the
          function, ``False`` if a fallback record was synthesised

    Raises
    ------
    TypeError
        If ``name`` is not a string.

    Examples
    --------
    >>> ex = sp.examples("did")
    >>> ex["examples"][0]["title"]
    'Classic 2x2 DID'

    See Also
    --------
    sp.describe_function :
        Full registry record (params / failure_modes / etc.).
    sp.list_functions :
        Discover available function names.
    sp.recommend :
        Method advisor when you don't yet know which function to call.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a string; got {type(name).__name__}")

    key = name.strip().lower()
    if not key:
        raise ValueError("name must be non-empty")

    # Look up the registry entry. Catch any registry import / lookup
    # failure rather than letting it propagate — agents often probe
    # examples speculatively.
    record: Dict[str, Any] = {}
    known = False
    try:
        from ..registry import describe_function as _describe
        try:
            record = _describe(key) or {}
            known = True
        except Exception:
            record = {}
    except Exception:
        record = {}

    snippets = list(_CURATED.get(key, []))
    # If no curated snippet but the registry has an `example` field,
    # surface that as a single snippet so the agent gets at least one.
    if not snippets:
        registry_example = (record.get("example") or "").strip()
        if registry_example:
            snippets.append({
                "title": "Registry quick-start",
                "code": (f"import statspai as sp\n"
                         f"# (replace df / column names with your data)\n"
                         f"{registry_example}"),
            })

    return {
        "name": key,
        "category": record.get("category"),
        "description": record.get("description", ""),
        "signature": record.get("example") or None,
        "examples": snippets,
        "pre_conditions": list(record.get("pre_conditions") or []),
        "assumptions": list(record.get("assumptions") or []),
        "alternatives": list(record.get("alternatives") or []),
        "known_function": known,
    }


__all__ = ["examples"]
