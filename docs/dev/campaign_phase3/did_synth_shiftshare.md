# Phase 3 — did_synth / `shiftshare` cluster

Functions: `sp.shift_share_se`, `sp.bartik`, `sp.ssaggregate`.
Worktree: `.claude/worktrees/pc-did-synth`. All changes uncommitted.

Data: `tests/reference_parity/_fixtures/did_synth_shiftshare_loc.csv` and
`did_synth_shiftshare_shocks.csv`, written by
`_fixtures/_generate_did_synth_shiftshare_data.py` (seed 20260918). 240 locations
in 12 states x 20 industries. Shares are Dirichlet(0.5) scaled by a
location-specific total in [0.55, 0.95], so they do **not** sum to one; `ssum` is
the sum-of-shares control. Errors carry an industry component (`sh @ nu`) and a
state component; `x` is endogenous. The Bartik column `B = sh @ g` is in the CSV,
so every side reads the same bytes.

Generators:

- R: `tests/reference_parity/_generate_did_synth_shiftshare_R.R` writes
  `_fixtures/did_synth_shiftshare_R.json` (R 4.5.2; ShiftShareSE 1.1.0, AER 1.2.16,
  sandwich 3.1.1, fixest 0.14.0, data.table 1.18.2.1, ssaggregate 0.0.0.9000
  (GitHub kylebutts/ssaggregate@22df93980250891a0cc247f6020136cd33c65ba2),
  bartik.weight 0.1.0 (GitHub paulgp/bartik-weight@722ceb85484d6a2bf77985edf2403515eacd1770,
  `R-code/pkg`)). Versions are recorded in the JSON.
- Stata: `tests/reference_parity/_fixtures/_generate_did_synth_shiftshare_stata.do`
  writes `_fixtures/did_synth_shiftshare_stata.json` (Stata 18 MP; SSC `reg_ss` /
  `ivreg_ss` 20241116, `ssaggregate` 1.2.2 (20200826), `ivreg2`; `bartik_weight.ado`
  from paulgp/bartik-weight@722ceb8 `code/`).

Test: `tests/reference_parity/test_did_synth_shiftshare_parity.py` (24 tests, all
passing). Tolerance is 1e-9 relative everywhere. p-values also get an absolute floor
of 1e-15, because both references compute `2 * (1 - Phi(|t|))` and StatsPAI uses the
survival function. At p ~ 1e-12 that cancellation alone gives a 8.5e-5 relative gap,
which is the reference's rounding error and not a disagreement.

## 1. Per-function table

| function | reference (version) | outcome class | max rel err est / SE | test file |
| --- | --- | --- | --- | --- |
| `sp.ssaggregate` (AKM inference: IV and OLS modes, with and without controls; rows Homoscedastic / EHW / Reg. cluster / AKM / AKM0; AKM / AKM0 CIs and p-values; alpha = 0.10) | R `ShiftShareSE::ivreg_ss` / `reg_ss` 1.1.0; Stata SSC `ivreg_ss` / `reg_ss` 20241116 | **2** (defect fixed) → 1 | R 8.8e-15 / 1.2e-14; Stata 2.4e-14 / 6.0e-14 | `test_did_synth_shiftshare_parity.py` |
| `sp.ssaggregate` (BHJ shock-level aggregation `s_n`, `ybar_n`, `xbar_n` plus shock-level IV and its HC0 SE) | R `ssaggregate` (kylebutts@22df939) + `AER::ivreg` / `sandwich` HC0; Stata SSC `ssaggregate` 1.2.2 + `ivreg2 ..., robust` | **2** (missing feature: the function never aggregated) → 1 | R beta 1.1e-16 / SE 2.0e-15 / frame 2.9e-13; Stata 1.2e-16 / 1.8e-15 / 9.3e-14 | same |
| `sp.shift_share_se` (on an `sp.bartik` fit, with and without controls) | R `ShiftShareSE::ivreg_ss` 1.1.0 (AKM row); Stata `ivreg_ss` | **2** (defect fixed) → 1 | R 1.5e-15 / 3.1e-15; Stata 1.8e-14 / 6.0e-14 | same |
| `sp.bartik` (2SLS coefficients; `robust='hc1'` and `'nonrobust'` SEs) | R `AER::ivreg` + `sandwich::vcovHC(type="HC1")` / classical; Stata `ivregress 2sls, vce(robust) small` / `, small` | **1** (numbers were already right) | R 1.7e-15 / 5.4e-15; Stata 1.8e-15 / 1.8e-15 | same |
| `sp.bartik` Rotemberg weights (`alpha_k`, `beta_k`) | R `bartik.weight::bw` (paulgp@722ceb8); Stata `bartik_weight` (paulgp@722ceb8) | **2** (weights unreachable from `sp.bartik`, `beta_k` missing) → 1 | alpha 2.2e-12 (R) / 2.4e-12 (Stata); beta 2.2e-12 / 2.5e-12 | same |

Reference-free identities in the same file:

- BHJ Prop. 1: with the sum-of-shares control, the shock-level coefficient equals the
  location-level shift-share IV (1e-12).
- With complete shares and only an intercept, the BHJ HC0 SE equals the AKM SE
  (1e-10).
- GPSS: `sum_k alpha_k beta_k` equals the 2SLS estimate, and `sum_k alpha_k = 1`.

The two references agree with each other on every shared quantity (R and Stata AKM
SE to about 1e-14), so these are T2 rows.

**Why the existing `test_bartik_ssagg_parity.py` never met a reference.** It only
checked the point estimate against `cov(B, y) / cov(B, x)`, which the old code got
right. `sp.ssaggregate` was a location-level 2SLS with its own variance formula, so
it produced no shock-level frame that Stata or R `ssaggregate` could be compared
with. Its SE, the part that was wrong, was never tested against anything.

## 2. Defects

### D1 — `sp.ssaggregate` AKM SE was the wrong formula, anti-conservative (⚠️ correctness)

- **What.** The code computed `u_k = sum_i s_ik * Ztilde_i * e_i` (instrument inside
  the sum) and divided by `(Xhat'Xtilde)^2`. AKM is `cR_k = hX_k * sum_i s_ik e_i`,
  where `hX` are the coefficients of the control-residualised shift-share variable on
  the shares (the control-adjusted shocks), with `RX = ddY2'ddX`.
- **How found.** On the fixture, the first divergence was the SE of `x` against
  `ShiftShareSE::ivreg_ss(method="all")`: 0.0665 against 0.2904. Reading
  `ivreg_ss.fit` / Stata `s_AKM1_nocluster_iv` gave the correct `cR_k`.
- **Fix.** A new private kernel, `src/statspai/bartik/_akm.py::_akm_fit`, ports
  `reg_ss.fit` / `ivreg_ss.fit` line by line, including `drop_collinear`, AKM0 CI
  inversion, the region-cluster row and the null-imposed p-value. `ssaggregate` now
  calls it. Separately, the old `SE (HC1)` diagnostic for `x` left out the
  first-stage `gamma` in the meat. It is now the HC1 SE from the full 2SLS sandwich,
  which equals `ivregress ..., vce(robust) small`.
- **Before → after (fixture, SE of `x`).**

  | spec | before | after (= R = Stata) |
  | --- | --- | --- |
  | IV + controls `c1 ssum` | 0.06646 | 0.29044 |
  | IV, intercept only | 0.04967 | 0.27941 |
  | OLS reduced form + controls (`shocks=None`) | 0.22565 | 0.96533 |
  | `SE (HC1)` diagnostic, IV + controls | 0.17750 | 0.27585 |

  **Default output changed: yes.** The SE of `x` and its p-value / CI change.
  `p-values` / CIs of all coefficients now use z instead of t(n−k)
  (`data_info["inference"] = "z"`), as both references do. Point estimates are
  unchanged.

### D2 — `sp.shift_share_se` used the second-stage fitted values as the instrument (⚠️ correctness)

- **What.** It took `Ztilde = fitted_values - mean` (for an `sp.bartik` fit that is
  `yhat`, not the instrument), applied the same wrong `u_k`, and used `Ztilde'Ztilde`
  as the denominator. For any IV result without stored inputs it silently returned
  these numbers.
- **How found.** Same comparison: 0.01759 against AKM's 0.29044 (16.5x too small).
- **Fix.** `sp.bartik` and `sp.ssaggregate` now store the outcome, endogenous
  regressor, shift-share instrument and control matrix in
  `data_info["_shift_share_inputs"]`. `shift_share_se` reruns `_akm_fit` on them and
  checks that the fitted coefficient is reproduced. Results without these inputs
  (for example `sp.regress`, `sp.iv`) now **raise `ValueError`** instead of returning
  a fabricated SE.
- **Before → after.** 0.017595 → 0.290442 (IV + controls). **Default output changed:
  yes**, and inputs that are not supported now raise.

### D3 — `sp.ssaggregate` did not do what its name says

It never produced the BHJ shock-level data set. It now attaches
`data_info["shock_data"]` (`shock`, `s_n`, `y`, `x`, `g`) and adds the diagnostics
`beta (BHJ shock-level)` / `SE (BHJ shock-level, HC0)`, equal to Stata / R
`ssaggregate` followed by `ivreg2 ..., robust`. Like both references, it warns when
the shares are incomplete and the sum of shares is not spanned by the controls. This
is additive: no existing number changes.

### D4 — silently ignored arguments

- `sp.ssaggregate(cluster=...)` was documented as "not used". It now drives the
  `SE (Reg. cluster)` row (as in `ShiftShareSE`) and raises if the column is missing.
- `sp.ssaggregate(alpha=...)` was ignored. It now sets the AKM / AKM0 CI level.
- `sp.bartik(robust=...)` treated any value other than `'nonrobust'` as HC1 (for
  example `robust='hc3'` silently gave HC1). Only `'hc1'` / `'robust'` and
  `'nonrobust'` / `'classical'` / `'unadjusted'` are accepted now; anything else
  raises `ValueError`. The numbers for supported values are unchanged.

### D5 — Rotemberg weights unreachable from `sp.bartik`

They were stored only on the `BartikIV` instance, which `sp.bartik` discards. They
are now in `result.model_info["rotemberg_weights"]`, with the GPSS just-identified
`beta_k` column that `bartik_weight` / `bw()` return. A zero denominator now warns
and reports NaN instead of silently returning all-zero weights. The alpha values
themselves were already correct.

## 3. Proposed promotion records (`scripts/build_parity_index.py::_FROZEN_PROMOTIONS`)

```python
    "ssaggregate": {
        "status": "bit-exact",
        "reference": (
            "R ShiftShareSE::ivreg_ss / reg_ss (Adao, Kolesar & Morales) and "
            "ssaggregate (Borusyak, Hull & Jaravel; R kylebutts/ssaggregate + "
            "AER::ivreg/sandwich HC0); Stata SSC ivreg_ss / reg_ss and "
            "ssaggregate + ivreg2, robust"
        ),
        "reference_versions": {
            "R": "4.5.2",
            "ShiftShareSE": "1.1.0",
            "ssaggregate (R)": (
                "0.0.0.9000 (GitHub kylebutts/ssaggregate@"
                "22df93980250891a0cc247f6020136cd33c65ba2)"
            ),
            "AER": "1.2.16",
            "sandwich": "3.1.1",
            "Stata": "18 MP",
            "reg_ss / ivreg_ss": "SSC 20241116",
            "ssaggregate (Stata)": "SSC 1.2.2 (20200826)",
        },
        "tolerance": (
            "1e-9 rel on beta, every SE row (Homoscedastic, EHW, Reg. cluster, "
            "AKM, AKM0), AKM/AKM0 CIs and the shock-level frame; p-values also "
            "atol 1e-15 (references use 2*(1-Phi)); observed <= 6e-14 "
            "(frame 2.9e-13)"
        ),
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_did_synth_shiftshare_parity.py",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_R.json",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_stata.json",
        ],
        "note": (
            "Incomplete shares (row sums 0.55-0.95), IV and OLS (reduced-form) "
            "modes, with and without controls, region clustering, AKM0 at "
            "alpha 0.05 and 0.10. Until this sweep the AKM SE used "
            "u_k = sum_i s_ik Z_i e_i instead of hX_k * s_k'e and was 4.4x too "
            "small on this data (0.0665 vs 0.2904); the BHJ shock-level "
            "aggregation did not exist. Regenerate via "
            "_generate_did_synth_shiftshare_R.R and "
            "_fixtures/_generate_did_synth_shiftshare_stata.do."
        ),
    },
    "shift_share_se": {
        "status": "bit-exact",
        "reference": (
            "R ShiftShareSE::ivreg_ss (Adao, Kolesar & Morales), AKM row; "
            "Stata SSC ivreg_ss"
        ),
        "reference_versions": {
            "R": "4.5.2",
            "ShiftShareSE": "1.1.0",
            "Stata": "18 MP",
            "ivreg_ss": "SSC 20241116",
        },
        "tolerance": "1e-9 rel on beta and the AKM / AKM0 / EHW / Homoscedastic SEs; observed <= 6e-14",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_did_synth_shiftshare_parity.py",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_R.json",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_stata.json",
        ],
        "note": (
            "Applied to an sp.bartik 2SLS fit with and without controls. Until "
            "this sweep it used the second-stage fitted values as the "
            "instrument and returned 0.0176 against AKM's 0.2904; it now "
            "raises on results that do not record the shift-share inputs."
        ),
    },
    "bartik": {
        "status": "bit-exact",
        "reference": (
            "2SLS: R AER::ivreg + sandwich (HC1 / classical), Stata ivregress "
            "2sls, vce(robust) small / small; Rotemberg weights: R "
            "bartik.weight::bw and Stata bartik_weight (Goldsmith-Pinkham, "
            "Sorkin & Swift)"
        ),
        "reference_versions": {
            "R": "4.5.2",
            "AER": "1.2.16",
            "sandwich": "3.1.1",
            "bartik.weight": (
                "0.1.0 (GitHub paulgp/bartik-weight@"
                "722ceb85484d6a2bf77985edf2403515eacd1770, R-code/pkg)"
            ),
            "Stata": "18 MP",
            "bartik_weight": (
                "GitHub paulgp/bartik-weight@722ceb85484d6a2bf77985edf2403515eacd1770 "
                "code/bartik_weight.ado"
            ),
        },
        "tolerance": (
            "1e-9 rel on all coefficients and SEs (observed <= 5.4e-15) and on "
            "Rotemberg alpha_k / beta_k (observed 2.5e-12)"
        ),
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_did_synth_shiftshare_parity.py",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_R.json",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_stata.json",
        ],
        "note": (
            "leave_one_out=False (the leave-one-out instrument has no "
            "reference). Rotemberg weights are exposed in "
            "model_info['rotemberg_weights'] with the per-industry "
            "just-identified beta_k; robust= now rejects values it does not "
            "implement instead of silently using HC1."
        ),
    },
```

## 4. Proposed CHANGELOG / MIGRATION

**⚠️ Correctness**

- `sp.ssaggregate`: the SE of the shift-share coefficient is now the
  Adão-Kolesár-Morales SE as defined by R `ShiftShareSE` / Stata `ivreg_ss`
  (`cR_k = hX_k s_k'e`). The old formula summed `s_ik Z_i e_i` and was
  anti-conservative, 4.4x too small on the parity fixture (0.0665 → 0.2904). The
  `SE (HC1)` diagnostic of `x` also dropped the first-stage coefficient. p-values and
  CIs now use the normal distribution, as the references do. Matches R and Stata to
  6e-14.
- `sp.shift_share_se`: it used the second-stage fitted values as the instrument
  (0.0176 → 0.2904 on the fixture). It now recomputes the AKM SE from the inputs that
  `sp.bartik` / `sp.ssaggregate` record, and raises `ValueError` for results that
  don't carry them (previously it returned a meaningless number).

**Added**

- `sp.ssaggregate` returns the Borusyak-Hull-Jaravel shock-level data set in
  `data_info["shock_data"]` and the shock-level IV coefficient / HC0 SE (identical to
  Stata / R `ssaggregate` followed by `ivreg2 ..., robust`). It also reports all
  ShiftShareSE rows (Homoscedastic, EHW, region cluster, AKM, AKM0 with the inverted
  CI).
- `sp.bartik` exposes Rotemberg weights in `model_info["rotemberg_weights"]`, adding
  the per-industry just-identified estimate `beta` (as `bartik_weight` does).

**Fixed**

- `sp.ssaggregate(cluster=, alpha=)` were silently ignored. `cluster` now gives the
  region-cluster row and `alpha` sets the CI level.
- `sp.bartik(robust=...)` silently used HC1 for any value other than `'nonrobust'`.
  Values it does not implement now raise.

**MIGRATION rows**

| Area | Old | New (matches R ShiftShareSE / Stata ivreg_ss) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.ssaggregate` SE of `x` | `sqrt(sum_k (sum_i s_ik Z_i e_i)^2) / (Xhat'X)^2` | AKM `sqrt(sum_k (hX_k s_k'e)^2) / RX` | — (not a documented quantity) |
| `sp.ssaggregate` p-values / CIs | t(n−k) | z | — |
| `sp.ssaggregate` `diagnostics["SE (HC1)"]` | HC1 without the first-stage coefficient | HC1 of the 2SLS sandwich (`ivregress, vce(robust) small`) | — |
| `sp.shift_share_se` | AKM-like formula on `fitted_values`; worked on any result | AKM on the recorded instrument; raises for results not from `sp.bartik` / `sp.ssaggregate` | — |
| `sp.bartik(robust=<other>)` | silently HC1 | `ValueError` | `robust="hc1"` |

## 5. Not closed / out of scope

- **`sp.bartik(leave_one_out=True, regional_shocks=...)`.** No reference implements
  that exact leave-one-out instrument, so it is not compared (class 6).
  `shift_share_se` on such a fit applies the AKM formula with the LOO instrument as
  `X`, which AKM's theory does not cover. Not flagged in code.
- **AKM `sector_cvar` (sector-clustered AKM) and regression weights.** The references
  support both; StatsPAI's public API exposes neither, so they are not tested. The
  kernel has no weights path.
- **AKM0 `beta0 != 0`.** Not reachable from the public API. The shared kernel is
  checked against R and Stata at beta0 = 1.5 (p-value), and the CI does not depend on
  beta0.
- **`sp.bartik(alpha=...)`.** It is not used anywhere in `fit()` (results always
  carry 95% CIs). This is the same class of silently ignored argument. Left as is and
  recorded here.
- **Found but not in my function list:** `sp.shift_share_political_panel`
  (`bartik/political.py` ~L1030–1077) uses **the same wrong AKM formula**
  (`u_k = sum s_k * Z_tilde * eps`, no `hX`) for `diagnostics['akm_se']` when
  `cluster='shock'`. It should be moved onto `_akm.py::_akm_fit`, but its FE-demeaned
  panel design needs its own reference check. The `sp.shift_share_political` module
  docstring says it "runs an AKM shock-level cluster SE", but the cross-section path
  reports the HC1 SE of `sp.bartik`. Both need a follow-up.
- **Doc bug outside my ownership:** `docs/guides/choosing_iv_estimator.md` §7 shows
  `sp.shift_share_se(r)` without the required `shares=` argument.
- `tests/reference_parity/test_bartik_ssagg_parity.py` (analytical only) still
  passes. Its docstring is now accurate, but it is superseded by the new file for
  grading.
- **Schema drift:** no public signature changed, only docstrings and return content.
  The integrator should still rerun `scripts/dump_schemas.py`, since docstring text
  feeds the function schemas.

## 6. .gitignore

- `tests/reference_parity/_fixtures/_ado_did_synth/`: private Stata ado dir (SSC
  `reg_ss`, `ivreg_ss`, `ssaggregate`, `ivreg2`, `ranktest`, `ftools`, `reghdfe`,
  `ivreghdfe`, `moremata`, plus `b/bartik_weight.ado`). This is shared with the other
  did_synth sub-clusters.
- The Stata batch log `_generate_did_synth_shiftshare_stata.log` lands in `_fixtures/`
  when the generator runs. I deleted it, but a `*.log` rule for `_fixtures/` would
  help.

## Files

- Changed: `src/statspai/bartik/adao_correction.py`, `src/statspai/bartik/shift_share.py`
- New: `src/statspai/bartik/_akm.py`,
  `tests/reference_parity/test_did_synth_shiftshare_parity.py`,
  `tests/reference_parity/_generate_did_synth_shiftshare_R.R`,
  `tests/reference_parity/_fixtures/_generate_did_synth_shiftshare_data.py`,
  `tests/reference_parity/_fixtures/_generate_did_synth_shiftshare_stata.do`,
  `tests/reference_parity/_fixtures/did_synth_shiftshare_{loc,shocks}.csv`,
  `tests/reference_parity/_fixtures/did_synth_shiftshare_{R,stata}.json`
- Tests run green: the new file (24), `test_bartik_ssagg_parity.py`, `test_bartik.py`,
  `test_transport_and_shiftshare.py`, `test_shift_share_political.py`,
  `test_estimator_input_hardening.py`, `test_iv_dispatcher.py`,
  `test_cov95_iv_init.py`, `test_aliases.py`, `test_late_bind_contracts.py`,
  `test_next_steps.py`, `test_tierD_p2_regression_ldv_analytic.py`,
  `test_citation_integrity.py`, `test_estimator_provenance_round3/6.py`,
  `test_methods_appendix.py`, `test_recommend_frontier_designs.py`,
  `test_registry_drift_repair.py`, `test_type_stub.py`, and the doctests of both
  changed modules.
