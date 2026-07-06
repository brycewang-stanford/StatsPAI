# FINDING — `sp.callaway_santanna` silent ATT=0.0 when `control_group="nevertreated"` has no never-treated units

> **STATUS: RESOLVED (2026-07-06).** Fixed in `callaway_santanna.py` with an
> up-front guard that raises `MethodIncompatibility` when
> `control_group="nevertreated"` and the panel has no never-treated units
> (NaN/inf `g` mirror the internal never-treated encoding). No previously-valid
> estimate moves; only the silent `0.0` path changes. §12 logged in CHANGELOG
> (⚠️ Correctness) + MIGRATION (#callaway-santanna-nevertreated-no-control).
> The Tier G guard now takes the raise branch. 247 callaway/cs tests green.

**Surfaced by:** `tests/tier_eg/test_did_robustness.py::test_cs_no_never_treated_control_documented`
(did Tier G module, 2026-07-06).

**Severity:** MEDIUM. A user gets a specific wrong number that reads as
"no effect," not an error — a §7 fail-loud violation.

## Repro

```python
import statspai as sp
import sys; sys.path.insert(0, "tests")
from tier_eg._helpers import make_staggered_did, coef

d = make_staggered_did(n_units=150, att=2.0, seed=3)
d_all = d[d["g"] != 0]  # every unit is eventually treated — no never-treated group
r = sp.callaway_santanna(d_all, y="y", g="g", t="time", i="id",
                         control_group="nevertreated")
print(coef(r))   # -> 0.0  (silent; zero warnings; true ATT is ~2.0)
```

Reproduced identically on seeds 3, 7, 11 — always `ATT = 0.0`, zero warnings.
The same data with `control_group="notyettreated"` returns a non-zero estimate
(~1.5; the last cohort is unidentified so it is attenuated but not zero).

## Root cause (diagnosis, not yet patched)

With `control_group="nevertreated"` and no never-treated units, every group×time
`ATT(g,t)` has an empty comparison group, so each cell is undefined and is being
filled with `0.0` rather than dropped/errored; the aggregation over these
zero cells then returns a headline `ATT = 0.0`.

## Contrast

The estimator *does* fail loudly on other degenerate inputs (missing id column →
`KeyError`; NaN rows → listwise drop + finite estimate). Only the
"requested control group is empty" case slips through as a silent zero.

## Suggested fix (maintainer's call — JOSS review in flight, §12)

In `callaway_santanna`, before aggregation, validate that the requested
`control_group` yields at least one valid comparison unit for at least one
`ATT(g,t)` cell; otherwise `raise MethodIncompatibility` with a message like
*"control_group='nevertreated' but the panel has no never-treated units; use
control_group='notyettreated' or add never-treated controls."* This mirrors the
existing loud-failure paths and does not move any currently-correct number
(the only affected output is the silent 0.0, which is wrong).

## Guard status

`test_cs_no_never_treated_control_documented` pins **both** acceptable outcomes
(a clean raise, or the current silent 0.0), so the gap is test-visible now and
the test flips to the raise branch automatically once the guard lands — no test
churn required for the fix.
