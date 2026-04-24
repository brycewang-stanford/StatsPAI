# Migrating between StatsPAI versions + from PyStataR

Internal version-to-version migrations are at the top; the long-form
`PyStataR → StatsPAI` migration follows below.

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
