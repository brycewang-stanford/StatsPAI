# StatsPAI Grammar — learn the keywords once

Stata's real genius is not that it has many commands — it's that its *grammar*
is consistent: `, vce(cluster id)`, `, robust`, `weights`, `margins`, `estat`
mean the same thing across two hundred commands. R is more fragmented, but each
package is internally self-consistent.

StatsPAI exposes **1,100+ flat functions** under a single `import statspai as sp`.
That breadth is a strength only if the grammar stays consistent — otherwise the
same idea spelled five different ways (`robust` / `vcov` / `vce` / `se_type`)
makes a thousand functions feel like a thousand dialects. This page is the
contract that keeps the grammar honest, and the two CI gates that enforce it.

> **The pitch vs. R:** in R you hand-assemble `sandwich` + `clubSandwich` +
> `fwildclusterboot` + `lmtest` to get a full SE menu, and each has its own
> calling convention. StatsPAI's goal is that the *entire* SE menu (cluster /
> wild bootstrap / CR2–CR3 / multiway / Conley) is the **same, consistently
> spelled parameter** on every estimator. We are not all the way there yet
> (see the matrix below) — and we track exactly how far, in code.

---

## 1. Canonical keyword vocabulary

The single source of truth is [`statspai/_house_style.py`](../../src/statspai/_house_style.py).
The canonical spellings, ratified 2026-06-30:

| Concept | Canonical | Accepted aliases (forwarded) | Notes |
|---|---|---|---|
| Standard-error / variance type | **`vce`** | `robust`, `vcov`, `se_type` | Matches Stata `vce()`. `vcov` stays a permanent alias on the `fixest` family (pyfixest/R parity). |
| Outcome / dependent variable | **`y`** | `outcome`, `depvar`, `dependent` | EconML-mirrored `Y` is kept verbatim on the CATE/forest family. |
| Treatment variable | **`treat`** | `treatment`, `treatvar` | EconML-mirrored `T`/`W` kept verbatim on the CATE/forest family. |
| Clustering variable | **`cluster`** | `cluster_var` | `clusters` (plural) is reserved for multiway helpers that take a *list*. |
| Observation weights | **`weights`** | `weight`, `sample_weight` | `w`/`W` are **not** weights (spatial weight matrix / GMM weighting matrix). |
| Input dataframe | **`data`** | `df`, `frame` | Already near-universal (486 uses). |

### False friends (deliberately *not* unified)

Some parameters look like an inference keyword but mean something orthogonal —
unifying them would be a correctness bug, not a consistency win:

- `w` / `W` — spatial **weight matrix** (spatial econometrics), **GMM weighting
  matrix**, do-calculus **node set**, network **adjacency matrix**.
- `cov_type` — random-effects **covariance structure** in mixed models
  (`'unstructured'`), not an SE type.
- `group` — a comparison/cohort group in decompositions and DiD, not clustering.

The lint excludes these via an explicit allowlist so its signal stays trustworthy.

### Accepting the canonical spelling today (additive, reversible)

Estimators opt in via the [`accepts_aliases`](../../src/statspai/_aliases.py)
decorator, which forwards the canonical spelling to the existing parameter
**without renaming it** — old call sites keep working unchanged:

```python
import statspai as sp

sp.regress("y ~ x", df, vce="hc1")      # canonical — accepted today
sp.regress("y ~ x", df, robust="hc1")   # existing spelling — unchanged
# coefficients and SEs are bit-identical between the two
```

During JSS review this is strictly additive (no defaults change, no warnings).
Post-review, parameters are renamed to the canonical spelling and the legacy
spelling deprecates on the normal `MIGRATION.md` schedule.

---

## 2. The SE / vcov menu — honest coverage

The reviewer's sharpest question: *is wild cluster bootstrap usable on any
estimator, or only on `feols`?* The honest answer, tracked in
[`scripts/se_menu_matrix.py`](../../scripts/se_menu_matrix.py):

| estimator | classical | hc_robust | cluster | twoway | cr2_cr3 | wild_cluster_boot | conley | jackknife |
|---|---|---|---|---|---|---|---|---|
| `feols` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hdfe_ols` | ✓ | · | ✓ | ✓ | · | ✓ | · | · |
| `fepois` | ✓ | ✓ | ✓ | ✓ | · | · | · | · |
| `feglm` | ✓ | ✓ | ✓ | ✓ | · | · | · | · |
| `regress` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ivreg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ppmlhdfe` | ✓ | ✓ | ✓ | · | · | · | · | · |
| `panel` | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| `callaway_santanna` | · | · | ✓ | · | · | · | · | · |
| `did` | · | · | ✓ | · | · | · | · | · |
| `dml` | ✓ | · | · | · | · | · | · | · |
| `rdrobust` | · | ✓ | ✓ | · | · | · | · | · |
| `synth` | · | · | · | · | · | · | · | ✓ |

**Legend:** ✓ native (estimator parameter, correct for its FE/IV/GLM structure)
· ○ standalone (correct, via an `sp.*` SE function reading stored design/residuals)
· ⚠ standalone-**unsafe** (a standalone function that re-parses the formula and
refits plain OLS — **wrong** for FE/IV) · · n/a.

What the matrix makes explicit today:

- **`regress` has a full, native SE menu** (8/8) — all the standalone SE
  options are now `vce=` parameters on `sp.regress` itself:

  ```python
  sp.regress("y ~ x", df, vce="CR2", cluster="firm")       # bias-reduced CR2
  sp.regress("y ~ x", df, vce="CR3", cluster="firm")       # bias-reduced CR3 (== jackknife)
  sp.regress("y ~ x", df, vce="wild", cluster="firm")      # WCR cluster bootstrap
  sp.regress("y ~ x", df, cluster=["firm","year"])          # CGM-2011 two-way
  sp.regress("y ~ x", df, vce="conley",                    # Conley spatial HAC
                 conley_lat="lat", conley_lon="lon", conley_cutoff=200.0)
  ```

  **Validated vs Stata `reghdfe vce(cluster firm year)`, `acreg ... spatial
  dist()`, and R `sandwich::vcovCL(HC2/3)`** to machine precision on the
  same 400-obs panel (see `tests/reference_parity/test_ols_se_external_parity.py`).
- **Wild cluster bootstrap is native on `feols`** via `vce="wild"` (and on the
  panel `hdfe_ols` path). It runs the WCR bootstrap on the FE-absorbed within
  design:

  ```python
  sp.feols("y ~ x | firm", data=df, vce="wild", cluster="firm")
  # point estimates = feols; p-values & CIs = wild cluster bootstrap
  ```

  **Externally validated against Stata `boottest`** (David Roodman's canonical
  implementation): on an identical 600-obs / 15-cluster panel the point estimate
  and CRV1 cluster SE match `reghdfe` to ~1e-9, and the wild p-value matches
  `boottest`'s exact 2¹⁵ Rademacher enumeration to Monte-Carlo error (0.0265 vs
  0.0263). See `tests/reference_parity/test_feols_wild_boottest_parity.py`.
- **`ivreg` has native wild via `vce="wild"`** — the **WRE** bootstrap
  (Davidson-MacKinnon 2010), the IV-correct procedure that resamples both the
  structural and reduced-form residuals and refits 2SLS:

  ```python
  sp.ivreg("y ~ w + (d ~ z1 + z2)", data=df, vce="wild", cluster="firm")
  # the endogenous coefficient gets a wild-bootstrap p-value & CI
  ```

  **Validated against Stata `boottest` after `ivreg2`** across strong-IV (p
  0.2016 vs 0.20155) and weak-IV (p 0.3415 vs 0.3412) regimes — the weak-IV
  case rules out the naive reduced form (which gives 0.426). Also validated for
  **two endogenous regressors** (p 0.2101/0.0151 vs boottest 0.2108/0.0141).
  See `tests/reference_parity/test_iv_wild_boottest_parity.py`.
- **`ivreg` has native two-way clustering** via `cluster=["a", "b"]` — the
  CGM (2011) inclusion-exclusion sandwich on the projected regressors, matching
  Stata `ivreg2, cluster(a b) small`.
- **`ivreg` has native bias-reduced cluster SEs** via `vce="CR2"` (Bell-
  McCaffrey) and `vce="CR3"` (== `vce="jackknife"`) — the Pustejovsky-Tipton
  (2018) adjustment on the projected 2SLS regressors, matching R
  `clubSandwich::vcovCR(ivreg, type=...)` to machine precision.
- **`ivreg` has native Conley spatial HAC** via
  `vce="conley", conley_lat=, conley_lon=, conley_cutoff=` — the spatial kernel
  on the projected 2SLS scores with Stata `acreg`'s planar distance (111 km/deg,
  `cos(lat)` longitude), matching `acreg ... spatial` exactly.
- **`ivreg`'s SE menu is now complete (8/8 native):** every cell — classical /
  HC / cluster / two-way / CR2-CR3 / wild bootstrap / Conley / jackknife — is a
  native, externally-validated option. No ⚠ cells remain in the whole matrix.
- **`feols` has native bias-reduced / spatial SEs** via `vce="CR2"`, `vce="CR3"`
  (== `vce="jackknife"`) and `vce="conley"` on the FE-absorbed within design.
  The within-transform's leverage adjustment reproduces R
  `clubSandwich::vcovCR(plm, model="within", type=...)` to machine precision, so
  the entire `feols` row is 8/8 native
  (`tests/reference_parity/test_feols_bias_reduced_parity.py`).
- **`panel(method="fe")` mirrors that menu.** The one-way entity fixed-effects
  estimator accepts `vce="CR2"/"CR3"/"jackknife"`, `vce="conley"` (with
  `conley_lat=/conley_lon=/conley_cutoff=`) and two-way clustering via
  `cluster=["a", "b"]`. Since OLS on the entity-demeaned design reproduces the
  linearmodels FE coefficients, the CR2/CR3 SEs equal the *identical*
  `clubSandwich::vcovCR(plm)` anchor as `feols`, and Conley / two-way match
  `sp.regress` on the hand-demeaned data (Stata `acreg` / `reghdfe`
  conventions) — see `tests/reference_parity/test_panel_bias_reduced_parity.py`.

  ```python
  sp.panel(df, "y ~ x", entity="firm", time="t", method="fe",
           vce="CR2", cluster="firm")                 # bias-reduced CR2
  sp.panel(df, "y ~ x", entity="firm", time="t", method="fe",
           cluster=["firm", "region"])                # two-way cluster
  ```

This is the gap the SE-menu wiring work closes, estimator by estimator. The
matrix is the scoreboard: the CI gate ratchets the **native** count up and the
**unsafe** count down, so coverage can only improve.

---

## 3. The CI gates

Two `--check` gates keep the grammar from drifting:

```bash
python scripts/signature_house_style.py --check   # keyword-spelling ratchet
python scripts/se_menu_matrix.py --check          # SE-menu coverage ratchet
```

- **`signature_house_style`** introspects every public callable and counts
  legacy-spelling sites per theme against a frozen baseline. A new function
  using `robust=` where `vce=` is canonical raises the count and fails the
  gate. False friends are excluded by allowlist.
- **`se_menu_matrix`** validates that every estimator and SE function it
  references still resolves under `sp.*` (drift guard), and ratchets the
  native-cell count up / unsafe-cell count down.

Both are **ratchets, not hard zeros**: the surface has historical drift, so the
gates block *regressions* while letting convergence land incrementally — the
counts only move in the improving direction.
