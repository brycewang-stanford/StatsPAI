"""Compute the schema-quality numbers reported in Table 9 of the JSS draft.

Run::

    python scripts/schema_quality.py

Outputs the function-level coverage and parameter-level quality
statistics over the live ``sp.list_functions()`` registry.

The numbers feed
``Paper-JSS/manuscript/sections/07-agent-eval.tex``,
Table~\\ref{tab:schema-coverage}. Re-running this script after a
release verifies that the table has not silently drifted from the
implementation.
"""

from __future__ import annotations

import statspai as sp

CURATED_KEYS = (
    "assumptions",
    "pre_conditions",
    "failure_modes",
    "limitations",
    "minimum_n",  # legacy spelling used in older manuscript notes
    "typical_n_min",
)


def main() -> None:
    names = sp.list_functions()
    n_total = len(names)

    n_with_desc = 0
    n_with_params = 0
    n_curated_card = 0
    n_param_total = 0
    n_param_desc = 0
    n_param_default = 0
    n_param_enum = 0

    for name in names:
        try:
            spec = sp.describe_function(name)
        except Exception:
            spec = None
        desc = (spec or {}).get("description", "") if isinstance(spec, dict) else ""
        if isinstance(desc, str) and len(desc.strip()) > 5:
            n_with_desc += 1

        sch = sp.function_schema(name)
        if isinstance(sch, dict):
            props = (sch.get("parameters", {}) or {}).get("properties", {}) or {}
            if props:
                n_with_params += 1
                n_param_total += len(props)
                for p in props.values():
                    if not isinstance(p, dict):
                        continue
                    pdesc = p.get("description")
                    if isinstance(pdesc, str) and pdesc.strip():
                        n_param_desc += 1
                    if "default" in p:
                        n_param_default += 1
                    if "enum" in p:
                        n_param_enum += 1

        try:
            card = sp.agent_card(name)
        except Exception:
            card = None
        if isinstance(card, dict) and any(card.get(k) for k in CURATED_KEYS):
            n_curated_card += 1

    def pct(num: int, denom: int) -> str:
        return f"{100 * num / denom:.1f}%" if denom else "--"

    print(f"Public surface (registered functions) : {n_total}")
    print(f"  with non-empty description          : {n_with_desc}")
    print(f"  with at least one typed parameter   : {n_with_params}")
    print(
        f"  with curated agent_card             : {n_curated_card}  "
        f"(at least one of {CURATED_KEYS})"
    )
    print()
    print(f"Total parameters across all schemas   : {n_param_total}")
    print(
        f"  with non-empty description          : {n_param_desc}  "
        f"({pct(n_param_desc, n_param_total)})"
    )
    print(
        f"  with explicit default value         : {n_param_default}  "
        f"({pct(n_param_default, n_param_total)})"
    )
    print(
        f"  with enumerated choices (enum)      : {n_param_enum}  "
        f"({pct(n_param_enum, n_param_total)})"
    )


if __name__ == "__main__":
    main()
