# Migrating between StatsPAI versions + from PyStataR

Internal version-to-version migrations are at the top; the long-form
`PyStataR → StatsPAI` migration follows below.

---

## v1.11 → v1.12 — `modelsummary` becomes a thin facade over `regtable`

The R-style `modelsummary()` previously shipped a ~700-line renderer
pipeline that re-implemented coefficient extraction, star formatting,
three-line table styling and every export format. PR-B/5b in v1.12
collapses it to a thin facade that translates R-flavoured kwargs and
forwards to `sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable` (book-tab
three-line, publication-quality star legend). A `DeprecationWarning` is
emitted on first use; plan to migrate to `sp.regtable(...)` directly
within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `stars={"*": 0.10, "**": 0.05, "***": 0.01}` | only the threshold *values* are kept; the symbol overrides are dropped (regtable's ladder is `*/**/***` by convention; use `regtable(notation='symbols')` for `†/‡/§`) |
| `se_type='brackets'` | downgraded to parens with `UserWarning`; use `show_ci=True` for `[lo, hi]` if you want brackets to convey actual information |
| `se_type='none'` | downgraded to parens with `UserWarning`; the SE row stays |
| Stat keys `nobs/r_squared/adj_r_squared/f_stat` | translated to regtable canonical (`N`/`r2`/`adj_r2`/`F`) |
| Stat keys `method`/`bandwidth`/`estimand` | silently dropped (modelsummary-only; build a custom `add_rows={}` if needed) |

`coefplot` is unchanged — independent of the table renderer.

### Side-by-side migration

```python
# Before — R-style functional API
sp.modelsummary(m1, m2, m3,
                model_names=["Base", "Mid", "Full"],
                stats=["nobs", "r_squared", "adj_r_squared"],
                output="latex")

# After — direct regtable call (same LaTeX output, full control)
sp.regtable(
    [m1, m2, m3],
    model_labels=["Base", "Mid", "Full"],
    stats=["N", "r2", "adj_r2"],
).to_latex()
```

---

## v1.11 → v1.12 — `outreg2` becomes a thin facade over `regtable`

The Stata-style `OutReg2` class and `outreg2()` function previously
shipped a bespoke 800-line renderer that re-implemented coefficient
extraction, star formatting, three-line table styling, and Excel /
Word / LaTeX export. PR-B in v1.12 collapses that to ~150 lines of
glue that translates Stata-flavoured kwargs and forwards to
`sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable`'s
canonical book-tab style. The visible label changes are listed below.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Label / format changes

| Legacy outreg2 output | New (regtable canonical) |
| --- | --- |
| `Variables` column header | blank (book-tab convention) |
| `R-squared` | `R²` |
| `Adj. R-squared` | `Adj. R²` |
| `Observations` | `N` |
| `F-statistic / Trees` | `F` *(bug fix: "/ Trees" only applied to causal-forest results)* |
| LaTeX missing star legend | proper `\multicolumn` legend below the rule |
| LaTeX `& None & None \\` junk row | gone *(bug fix: spurious empty ATE row)* |

### Removed parameter

| Old | New |
| --- | --- |
| `show_se=False` | no longer supported. Emits `UserWarning`; the SE row stays. Use `sp.regtable(..., se_type='t' \| 'p' \| 'ci')` directly if you need a different cell. |

### Side-by-side migration

```python
# Before — Stata-style stateful builder
o = sp.OutReg2()
o.set_title("Wage Regressions")
o.add_model(m1, "Baseline")
o.add_model(m2, "Full")
o.add_note("Robust SE in parentheses")
o.to_excel("table1.xlsx")

# After — direct regtable call (same Excel output, full control)
sp.regtable(
    [m1, m2],
    title="Wage Regressions",
    model_labels=["Baseline", "Full"],
    notes=["Robust SE in parentheses"],
).to_excel("table1.xlsx")
```

---

## Migrating from `pyreghdfe`

`pyreghdfe` (`pip install pyreghdfe`) is a Python port of Stata's
`reghdfe` maintained as a standalone package. Its scope — multi-way FE
OLS with robust / multi-way cluster SEs, singleton dropping, weighted
regression — is now a strict subset of `sp.hdfe_ols` / `sp.absorb_ols`
in StatsPAI.

### API mapping (pyreghdfe → StatsPAI)

| `pyreghdfe` | StatsPAI (`import statspai as sp`) |
| --- | --- |
| `reghdfe(data=df, y='y', x=['x'], fe=['firm','year'], cluster=['firm'])` | `sp.absorb_ols(y=df['y'].values, X=df[['x']].values, fe=df[['firm','year']], cluster=df['firm'].values, solver='lsmr')` |
| Stata-style formula via pyreghdfe is not supported | `sp.hdfe_ols("y ~ x \| firm + year", data=df, cluster="firm")` (formula interface via pyfixest backend) |
| `solver='lsmr'` / `'lsqr'` | `solver='lsmr'` / `'lsqr'` — same Krylov paths (scipy.sparse.linalg) |
| Krylov-based solvers (LSMR/LSQR) | default `solver='map'` — alternating projections + Irons-Tuck acceleration, typically faster on well-conditioned panels. LSMR/LSQR remain opt-in for pathological FE structures. |
| weighted regression | `weights=` kwarg; LSMR path uses the standard √w transformation on both the sparse design and the response |
| singleton drop | `drop_singletons=True` (default) |
| multi-way cluster SE | `cluster=[firm_arr, year_arr]` (inclusion-exclusion CGM with PSD correction) |

### What you also get

- `sp.ppmlhdfe` — Poisson pseudo-ML with HDFE (not available in `pyreghdfe`).
- Rust-accelerated mean-sweep kernel ([rust/statspai_hdfe/](rust/statspai_hdfe/)).
- Formula interface and unified result object (`summary()`, `to_latex()`, `to_excel()`).
- One-line cross-solver parity check (all three solvers exposed under the
  same API — see `tests/test_hdfe_native.py::test_demean_alt_solver_matches_map_two_way`).

### Numerical parity

Default MAP and `solver='lsmr'` / `'lsqr'` agree on identical data to
`atol=1e-6` on two-way FE OLS (with and without weights, with and
without clustering). See the cross-solver parity suite in
`tests/test_hdfe_native.py`. We do not take a runtime dependency on
`pyreghdfe`; correctness is anchored to scipy's battle-tested
`scipy.sparse.linalg.lsmr` / `lsqr` plus the internal MAP baseline.

### When to prefer which solver

- **Default (`solver='map'`)**: almost everything. MAP + Aitken is
  typically 2–5× faster than LSMR on canonical firm × year panels.
- **`solver='lsmr'`**: ill-conditioned / highly nested FE structures
  where MAP shows slow convergence (`converged=False`,
  `iters==maxiter`). LSMR is more robust to near-redundancy between FE
  dimensions.
- **`solver='lsqr'`**: exposed for users migrating from code that
  explicitly requested LSQR. For new work prefer LSMR, which scipy
  implements on the same interface and generally offers better
  numerical stability on sparse least-squares.

---

## v1.8.0 → v1.9.0 — Agent-native API surface (no breaking changes)

**Strictly additive release.** Twelve new agent-shaped APIs land
under ``sp.``: ``audit``, ``bib_for``, ``brief``, ``detect_design``,
``examples``, ``preflight``, ``session`` (the seven new top-level
functions), plus ``result.brief()`` / ``result.cite(format=...)``
methods, plus three MCP-server features (``statspai-mcp`` console
script, ``prompts/list``, per-function ``statspai://function/{name}``
resources). **No estimator numerical paths changed**; every
coefficient / SE / CI / p-value is byte-identical to v1.8.0. See
the v1.9.0 [CHANGELOG](CHANGELOG.md#190--agent-native-api-surface-12-modules-across-4-phases)
entry for the full surface.

### Backward-compat invariants the test suite pins

The 422 new tests include explicit regression guards on these
contracts. If your code depended on any of them, nothing changes.

- ``CausalResult.to_dict()`` with no kwargs is **byte-identical**
  to ``to_dict(detail="standard")`` — the legacy default. The new
  ``detail`` parameter is keyword-only and adds three documented
  levels (``"minimal"`` / ``"standard"`` / ``"agent"``).
- ``CausalResult.cite()`` with no kwargs still returns a BibTeX
  string. The new ``format=`` keyword adds ``"apa"`` / ``"json"``
  options without changing the default.
- ``result.for_agent()`` is now a thin alias for
  ``result.to_dict(detail="agent")`` and produces the same dict.
  Existing callers see no change; new code should prefer the
  explicit form for readability.
- ``result.to_agent_summary()`` is unchanged. Its docstring now
  cross-references ``to_dict(detail="agent")`` so future readers
  know the distinction (``to_agent_summary`` is the *nested*
  schema with a ``point`` sub-dict; ``to_dict(detail="agent")`` is
  the *flat* schema). Both round-trip through ``json.dumps``.
- ``execute_tool``'s exception envelope still carries the legacy
  ``error`` / ``tool`` / ``arguments`` / ``remediation`` fields
  unchanged. Two new fields — ``error_kind`` and ``error_payload``
  — are added **only** when the caught exception is a
  ``StatsPAIError`` subclass, so any agent that previously branched
  on ``"error_kind" in out`` to detect structured errors gets a
  clean signal.

### One subtle widening to be aware of

- ``sp.agent.execute_tool``'s default serializer now invokes
  ``r.to_dict(detail="agent")`` instead of ``r.to_dict()``. The
  result dict is a strict superset of the previous shape — every
  pre-1.9 key is still present at the same path; ``violations``,
  ``warnings``, ``next_steps``, and ``suggested_functions`` are
  added. The MCP ``tools/call`` payload is therefore ~3× larger by
  default. Agents that need the smaller form should pass
  ``detail="standard"`` (or ``"minimal"``) in the ``tools/call``
  arguments — the MCP input schema documents this.

### New entry points worth knowing about

- Agents handed unfamiliar data → ``sp.detect_design(df)``.
- Before an expensive call → ``sp.preflight(df, "did", y=..., ...)``.
- After fitting → ``result.brief()`` for dashboards,
  ``sp.audit(result)`` for the missing-evidence checklist,
  ``result.cite(format="apa")`` for prose citations.
- Reproducible RNG → ``with sp.session(seed=42): ...``.
- One-shot install for MCP clients → ``pip install statspai`` now
  exposes ``statspai-mcp`` on PATH (Claude Desktop /
  ``claude_desktop_config.json`` example in
  [agent/mcp_server.py](src/statspai/agent/mcp_server.py)).

---

## v1.6.5 → v1.6.6 — ⚠️ Heckman two-step SE correctness fix (+ HDFE solver option)

**Two-part release.** (1) Correctness fix for `sp.heckman` standard
errors — point estimates unchanged, **SE / t / p / CI change**.
(2) Additive HDFE LSMR/LSQR solver option — all HDFE MAP output is
byte-identical to v1.6.5.

### What changed numerically (Heckman two-step)

`sp.heckman(...)` previously reported an HC1-style sandwich that the
source code itself flagged as
`"Heckman SEs are complex; robust is conservative"`. This was a known
limitation, not a secret bug — but it meant reported SEs, t-stats,
p-values and CIs were off by an amount that depended on (a) how
strongly selection induced heteroskedasticity `σ²(1 − ρ² δ_i)` and
(b) how uncertain the probit first-stage estimate γ̂ was.

v1.6.6 replaces it with the textbook Heckman (1979) / Greene (2003, eq.
22-22) / Wooldridge (2010, §19.6) analytical two-step variance:

```text
V(β̂) = σ̂² (X*'X*)⁻¹ [ X*'(I − ρ̂² D_δ) X* + ρ̂² F V̂_γ F' ] (X*'X*)⁻¹
```

- `X*`: second-stage design matrix including λ̂ as its last column.
- `δ_i = λ̂_i (λ̂_i + Z_iγ̂) ≥ 0` (Mills' ratio inequality).
- `D_δ = diag(δ_i)`; `F = X*' D_δ Z` (`k × q`).
- `V̂_γ = (Z' diag(w_i) Z)⁻¹` with probit information weights
  `w_i = φ(Z_iγ̂)² / [Φ(Z_iγ̂)(1 − Φ(Z_iγ̂))]`.
- `σ̂² = RSS / n_sel + β̂_λ² · mean(δ_i)` (Greene 22-21) —
  replaces the old naive `RSS / (n_sel − k)`.
- `ρ̂² = β̂_λ² / σ̂²`.

`model_info['sigma']` / `model_info['rho']` now also use this
consistent σ̂², so downstream code reading those fields will see
slightly different numbers.

### Who is affected

- Any caller of `sp.heckman(...)` — SEs, t-stats, p-values, CIs change.
- Point estimates `β̂` **do not change** (OLS of y on [X, λ̂]
  is unaffected by the variance formula).
- Callers that pin SE values in their own test suites against a
  pre-v1.6.6 StatsPAI will need to re-baseline.

### What you should do

1. **If you cited a Heckman SE / t / p / CI from StatsPAI ≤ 1.6.5**,
   re-run and update. The direction of change depends on whether
   selection-induced heteroskedasticity (reduces SE) or
   generated-regressor uncertainty (increases SE) dominates.
2. **Cross-validation**: compare the new output against Stata
   `heckman y x, select(z) twostep` or R
   `sampleSelection::heckit(...)`. Both implement the same Heckman
   (1979) formula; agreement should be to the documented precision.
3. **If you want the old conservative HC1 sandwich** for any reason
   (e.g. replicating a legacy pipeline), there is no supported way to
   get it. The old formula was not a convention choice — it was a
   known approximation the project had not yet replaced.

### Reference formula

Same as above, with the influence-function derivation:

```text
β̂ − β = (X*'X*)⁻¹ [ X*' e − β̂_λ · X*' D_δ Z · (γ̂ − γ) ] + o_p(n^{-1/2})
```

The first term gives the heteroskedastic `X*'(I − ρ̂² D_δ) X*`
contribution; the second gives the `ρ̂² F V̂_γ F'` generated-regressor
contribution, since `∂λ / ∂γ' = −λ(λ + Zγ) Z' = −δ · Z'`.

---

## v1.6.4 → v1.6.5 — ⚠️ Standalone LIML correctness fix

**Narrow correctness follow-up to v1.6.4.** If your codebase only uses
`sp.ivreg`, `sp.iv.iv`, `sp.iv.fit`, or `sp.ivreg(method='liml')` you
are **not affected** — those paths were fixed in v1.6.4. This release
closes an orphan copy of the same bug that lived in the standalone
`sp.liml` / `sp.iv.liml` entry point.

### What changed numerically

Anything calling `sp.liml(...)` directly will see both **β̂ and SE
change** compared to ≤ v1.6.4. Two independent bugs were fixed:

1. **κ_LIML solver**: switched from the non-symmetric
   `np.linalg.eigvals(inv(A) @ B)` (which can silently return complex
   eigenvalues and a biased κ) to the proper generalized symmetric
   eigenvalue problem `scipy.linalg.eigh(S_exog, S_full)`. Point
   estimates β̂ shift to the correct κ.
2. **Sandwich meat**: the cluster / robust meat used raw `X` instead of
   the k-class transformed `AX = (I − κ M_Z) X`. Same bug family as
   v1.6.4 for 2SLS; same fix (use the influence-function regressor in
   the meat).

### Post-fix consistency checks

- `sp.liml(...)` now produces **byte-identical** output to
  `sp.ivreg(..., method='liml')`.
- β̂ agrees with `linearmodels.IVLIML` to machine precision.
- Cluster SEs differ from `linearmodels.IVLIML` by ~0.1–0.2% because
  StatsPAI uses the k-class FOC-derived meat `AX = (I − κ M_Z) X`,
  while `linearmodels` uses the 2SLS-style meat `X̂ = P_Z X`
  regardless of κ. Both estimators are asymptotically equivalent and
  coincide exactly at κ = 1 (2SLS). The convention is documented in
  the new test file `tests/reference_parity/test_liml_se_parity.py`.

### What you should do

1. **If you have published LIML results** from a version ≤ v1.6.4 via
   `sp.liml(...)`, re-run and update — the old κ could be materially
   off and the old SE was built from the wrong meat.
2. **If you want LIML and only used `sp.ivreg(method='liml')`**, no
   action needed; v1.6.4 already has the correct formula.
3. **If you pinned SE or coefficient values** against the standalone
   `sp.liml` in your test suite, re-baseline to the v1.6.5 numbers.

### Reference formula (same as v1.6.4 for the k-class meat)

```text
β̂ − β = (X' A X)⁻¹ (AX)' u ,  A = (1 − κ) I + κ P_Z
Meat (cluster):  Σ_c (Σ_{i∈c} (AX)_i u_i)(·)'
Bread         :  (X' A X)⁻¹  = (AX' X)⁻¹
```

For 2SLS (κ = 1) `AX = P_Z X = X̂`; for LIML/Fuller `AX` is the
k-class transformed regressor.

---

## v1.6.3 → v1.6.4 — ⚠️ IV SE correctness fix

**Correctness-fix release.** No API surface changes, no new functions,
no docstring renames. **Numerical output of IV cluster / robust SE
changes** — this is the whole point of the release.

### What changed numerically

`sp.iv`, `sp.ivreg`, and `sp.iv.fit(method='2sls' | 'liml' | 'fuller')`
produce different standard errors when called with `robust={'hc0',
'hc1', 'hc2', 'hc3'}` or `cluster=...`. The fix restores the textbook
Cameron–Miller (2015) / Stata `ivregress` / `linearmodels` formula —
meat uses the projected regressor `X̂ = P_W X` rather than the raw
`X = [X_exog, X_endog]`.

Concretely the sandwich is now

```text
V̂ = (X̂'X̂)⁻¹ · [ Σ_c (X̂_c' û_c)(û_c' X̂_c) ] · (X̂'X̂)⁻¹
```

for the cluster case, and analogously for HC0/HC1/HC2/HC3. Before v1.6.4
the bread used `X̂` but the meat used `X`, which is a strictly incorrect
estimator for 2SLS — it happens to coincide with the correct formula
only when the first stage is a perfect fit (never, in practice).

### Who is affected

- Any IV workflow using `robust=` or `cluster=` with 2SLS, LIML, or Fuller.
- **Not affected**: point estimates (`β̂` is algebraically unchanged by
  the projection in the meat), nonrobust default SE, `method='gmm'`,
  `method='jive'`, and `sp.iv.ujive` / `ijive` / `rjive`.

### What you should do

1. **If you have published results** citing an IV SE / t-stat / p-value
   / CI from StatsPAI ≤ 1.6.3, re-run and update. The bias in the
   reported SE can be several-fold depending on first-stage fit —
   **not a rounding issue**.
2. **If you have pinned SE values in your test suite** against an
   earlier StatsPAI version, expect a mismatch. You can verify the new
   numbers by cross-checking with `linearmodels.IV2SLS(...).fit(
   cov_type='clustered', debiased=True)` — they should now agree to
   machine precision.
3. **If you were intentionally trying to reproduce the old (wrong)
   numbers**, don't. There is no supported way to get the
   pre-v1.6.4 behaviour because it was not a convention choice — it
   was a bug.

### Reference formula

For k-class with parameter κ (2SLS → κ=1, LIML → κ=κ_LIML, Fuller →
κ_LIML − α/(n−K)):

- Bread: `(X' A X)⁻¹` with `A = (1−κ) I + κ P_W`
- Meat: uses `A X` (the k-class transformed regressor); for 2SLS
  `A X = P_W X = X̂`
- FOC: `X' A (y − X β) = 0`, so the influence function is
  `β̂ − β = (X'AX)⁻¹ (AX)' u`, and the cluster/robust variance
  plugs `(AX)_i u_i` into the moment sum.

Pre-v1.6.4 the implementation plugged `X_i u_i` instead of `(AX)_i u_i`.

---

## v1.6.2 → v1.6.3 — DiD frontier sprint

**Strictly additive** plus one docstring / label truth-up. No existing
estimator's numerical path changes.

### User-visible changes worth noting

1. **`sp.continuous_did(method='att_gt')` result labels** —
   - ``result.method`` changed from
     `"Continuous DID (Callaway et al. 2024)"` to
     `"Continuous DID (dose-bin heuristic)"`.
   - ``result.estimand`` changed from
     `"ACRT (Average Causal Response on Treated)"` to
     `"Sample-weighted mean of dose-bin 2x2 DIDs (not CGS 2024 ATT(d|g,t))"`.
   - Why: the previous labels claimed paper fidelity with CGS (2024)
     that the implementation did not deliver. Numerical output is
     unchanged. If you were parsing these strings in a pipeline, update
     the matcher.
   - If you actually want a CGS (2024)-style estimator: the new
     `method='cgs'` is an **MVP** (2-period design, OR only) with
     paper formulas flagged `[待核验]`. See
     `docs/rfc/continuous_did_cgs.md`.

2. **`sp.did_multiplegt(dynamic=H)` semantic clarification** — the
   docstring now states explicitly that this is a pair-rollup
   extension, **not** the dCDH (2024) `did_multiplegt_dyn` estimator.
   Numerical output is unchanged; if you were using `dynamic=H` and
   calling it "dCDH 2024", switch to the new `sp.did_multiplegt_dyn`
   (also MVP — see `docs/rfc/multiplegt_dyn.md`).

### New functions (no migration needed, just additive)

`sp.lp_did`, `sp.ddd_heterogeneous`, `sp.did_timevarying_covariates`,
`sp.did_multiplegt_dyn` (MVP), `sp.continuous_did(method='cgs')` (MVP).

### Bib key updates

`paper.bib` entry `dechaisemartin2022fixed` upgraded from SSRN to the
published *Econometrics Journal* 26(3):C1–C30 (2023) version. Any
downstream uses of the bib key via `[@dechaisemartin2022fixed]` are
unaffected; the expanded citation will now render to the journal
version.

---

## v1.5.x → agent-native infrastructure (Unreleased)

Pure-additive release. **No migration required** for existing code.
New agent-native surface area documented here for adopters.

### 1. Exception taxonomy (new public module)

```python
from statspai.exceptions import (
    AssumptionViolation, IdentificationFailure,
    DataInsufficient, ConvergenceFailure,
    NumericalInstability, MethodIncompatibility,
)
```

Domain errors subclass the right stdlib base (`ValueError` /
`RuntimeError`), so existing `try / except ValueError` blocks still
catch `AssumptionViolation` and `DataInsufficient`, and
`except RuntimeError` still catches `ConvergenceFailure` and
`NumericalInstability`. No call-site changes required.

New code should prefer the specific subclass + attach a
`recovery_hint`:

```python
raise AssumptionViolation(
    "Parallel trends rejected at p=0.003",
    recovery_hint="Run sp.sensitivity_rr for Rambachan-Roth honest CI.",
    diagnostics={"test": "pretrends", "pvalue": 0.003},
    alternative_functions=["sp.sensitivity_rr", "sp.callaway_santanna"],
)
```

### 2. Agent-native result methods

- `result.violations()` — structured list of assumption /
  diagnostic issues with `severity` / `recovery_hint` / `alternatives`.
- `result.to_agent_summary()` — JSON-ready structured payload.
- Complement (do not replace) existing `summary()` / `tidy()` /
  `next_steps()`.

### 3. Registry agent cards

- `sp.agent_card(name)` — full metadata including pre-conditions,
  assumptions, failure modes with recovery hints, ranked
  alternatives, typical minimum N.
- `sp.agent_cards(category=None)` — bulk export of entries that
  have at least one agent-native field populated (currently:
  `regress`, `iv`, `did`, `callaway_santanna`, `rdrobust`, `synth`).

### 4. Guide `## For Agents` blocks

Run `python scripts/sync_agent_blocks.py` after any change to a
registered spec's agent-native fields. The `--check` flag is
CI-friendly and fails non-zero on drift.

---

## v1.4.x → v1.5.0

Minor release.  Only one change requires any migration:

### `sp.mr` is now a dispatcher function, not a module alias

Before v1.5.0, `sp.mr` was a reference to the `statspai.mendelian`
submodule, and `sp.mr.mr_ivw(...)` worked as attribute access on the
module.

In v1.5.0, `sp.mr` is the new **unified dispatcher** for the MR family,
matching the pattern of `sp.synth` / `sp.decompose` / `sp.dml`:

```python
sp.mr("ivw",   beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("egger", beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("mvmr",  snp_associations=snp_df,
       outcome="beta_y", outcome_se="se_y",
       exposures=["beta_bmi", "beta_ldl"])
```

| Old (<= v1.4.2) | New (>= v1.5.0) |
| --- | --- |
| `sp.mr.mr_ivw(...)` | `sp.mr_ivw(...)` (already available since v0.9) or `sp.mr("ivw", ...)` |
| `sp.mr.mr_egger(...)` | `sp.mr_egger(...)` or `sp.mr("egger", ...)` |
| `sp.mr.mr_presso(...)` | `sp.mr_presso(...)` or `sp.mr("presso", ...)` |
| `sp.mr` (as module alias) | `sp.mendelian` (module access preserved under this name) |

**Rule of thumb:** if your code uses `sp.mr_*` (underscore form) it
already works unchanged in v1.5.0.  Only the uncommon
`sp.mr.<attribute>` pattern needs rewriting.

### Output numerical differences you may notice after upgrading

- `sp.mr_egger` / `sp.mendelian_randomization(..., methods=["egger"])`
  slope p-values and CIs now use `t(n − 2)` rather than `Normal`, matching
  `sp.mr_pleiotropy_egger` and R's `MendelianRandomization` package.
  Effect is invisible for `n_snps ≥ ~100`.  For very small `n_snps` (say
  5 or 6) CIs widen by ~1.6×.
- `sp.mr_presso` p-values now use the `(k + 1) / (B + 1)` MC convention,
  so they are strictly positive (floor `1 / (B + 1)`).  No change for
  non-extreme cases; fixes `-inf` propagation through `log(p)` downstream.

---

## From PyStataR to StatsPAI

`PyStataR` is deprecated. All of its functionality is now available in
[StatsPAI](https://github.com/brycewang-stanford/StatsPAI), under a
unified `sp.*` namespace.

```bash
pip install statspai
```

```python
import statspai as sp
```

## API mapping

| PyStataR | StatsPAI |
|---|---|
| `pdtab.tab1(df, 'x')` / `tab2(df, 'x', 'y')` | `sp.tab(df, 'x')` / `sp.tab(df, 'x', 'y')` |
| `pywinsor2.winsor2(df, ['x'], cuts=(1,99))` | `sp.winsor(df, ['x'], cuts=(1,99))` |
| `pywinsor2.outlier_indicator(df, ['x'])` | `sp.outlier_indicator(df, ['x'])` |
| `pyoutreg.outreg(models, 'out.xlsx')` | `sp.outreg2(models, filename='out.xlsx')` |
| `pyegen.rowmean(df, ['x1','x2'])` | `sp.rowmean(df, ['x1','x2'])` |
| `pyegen.rowtotal(df, ['x1','x2'])` | `sp.rowtotal(df, ['x1','x2'])` |
| `pyegen.rowmax/rowmin(df, [...])` | `sp.rowmax(df, [...])` / `sp.rowmin(df, [...])` |
| `pyegen.rowsd(df, [...])` | `sp.rowsd(df, [...])` |
| `pyegen.rownonmiss(df, [...])` | `sp.rowcount(df, [...])` |
| `pyegen.rank(df, 'x', by='g')` | `sp.rank(df, 'x', by='g')` |

## Why migrate

- **One package, one namespace.** `sp.*` covers everything PyStataR did,
  plus DID, RD, synthetic control, IV, matching, DML, causal forest,
  meta-learners, and more.
- **Actively maintained.** PyStataR is frozen; new features land only in
  StatsPAI.
- **Cleaner naming.** No "Stata" in the name — StatsPAI is Python-native.

## Questions

Open an issue on
[StatsPAI/issues](https://github.com/brycewang-stanford/StatsPAI/issues).
