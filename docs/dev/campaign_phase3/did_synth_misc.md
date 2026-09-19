# Parity campaign phase 3 — did_synth / `misc` cluster

Functions: `sp.spillover_did`, `sp.harvest_did`, `sp.causal_impact`.

Files added (all uncommitted, worktree `pc-did-synth`):

- `tests/reference_parity/_generate_did_synth_misc_data.py` → `_fixtures/did_synth_misc_{harvest,spill_single,spill_stag,impact}.csv` (fixed seeds, `%.17g`)
- `tests/reference_parity/_generate_did_synth_misc_R.R` → `_fixtures/did_synth_misc_R.json`
  (R 4.5.2, did 2.3.0, DRDID 1.2.3, fixest 0.14.0, KFAS 1.6.0 (newly installed from CRAN), CausalImpact 1.4.1, bsts 0.9.11; recorded in `meta`)
- `tests/reference_parity/test_did_synth_misc_parity.py` (13 tests, all pass)

Files changed in `src/`: `did/harvest.py`, `did/spillover_rings.py`, `causal_impact/impact.py`, `causal_impact/__init__.py`.

Tests run after the change: the new file, plus every test file that references the three functions
(`test_spillover_rings.py`, `test_causal_impact.py`, `test_harvest_did.py`, `test_cov95_did_r3_harvest.py`,
`test_cov95_did_estimators*.py`, `test_cov95_synth_*`, `test_counterfactual_plot.py`, `test_estimator_provenance_round{3,6}.py`,
`test_late_bind_contracts.py`, `test_limitations_consistency.py`, `test_methods_appendix.py`, `test_synth_new_methods.py`,
`test_untested_public_api.py`): 585 passed. Doctests of the three modules pass.

## 1. Per-function table

| function | reference | class | max rel err est / SE | test |
| --- | --- | --- | --- | --- |
| `spillover_did` | did::att_gt(control_group="nevertreated") + did::aggte(type="simple"), one call per group (direct / ring r) on {group} ∪ {clean controls}, ring cohort = exposure onset (did 2.3.0); single cohort also fixest::feols(dbar ~ treat + ring1 + ring2, vcov="hetero", ssc(adj=FALSE)) = HC0 (fixest 0.14.0) | **2** (defect fixed, then T2) | did: 1.0e-14 / 6.5e-15; feols: 8.1e-16 / 3.2e-15 | `tests/reference_parity/test_did_synth_misc_parity.py` |
| `harvest_did` | cells: did::att_gt(control_group="notyettreated", base_period="universal"); event study with `weighting="n_treated"`: did::aggte(type="dynamic") (did 2.3.0) | **2** (defects fixed, then T2 for cells and the n_treated event study); the default `precision` aggregate is **6** (no reference defines it) | cells 1.7e-14 / 2.4e-15; event study 3.0e-15 / 4.1e-15 | same |
| `causal_impact` | R CausalImpact 1.4.1 / bsts 0.9.11 computes a **different model** (class 6). What sp actually runs is pinned: plug-in parameters vs lm/cor/sd, Kalman filter + forecast vs KFAS::KFS 1.6.0 | **6** vs CausalImpact; filter step T2 (3.7e-15 / 4.8e-15); SE defect fixed and pinned by a dense-Gaussian identity | — | same |

### spillover_did — reference search

- The module/test claim "no CRAN or GitHub package implements this estimator" is **true as a package** (checked: `did2s` has no ring/spillover function; Butts's GitHub has only replication repos). The replication code
  `github.com/kylebutts/Spatial-Spillover` (commit `d9e7b7b27ee78fc000cf78449f72ba9f4019885b`) shows the estimator is
  (a) `fixest::feols(d_y1 ~ treat + ring dummies)` on first differences (`code/rings-example/rings_example.R`), and
  (b) in staggered applications (`code/CHC/analysis.R`) a spillover indicator that switches on at the **first year a treated unit is within the distance band** (`year_within_25 <- min_year(...)`, `spill_rel_year`), estimated with `did2s`.
  `github.com/kylebutts/Difference-in-Differences-Ring-Method` (commit `5662915172094a68b97b02298ad70e93ae10a7eb`) is the separate geocoded-microdata rings paper (binsreg-based), a different estimator.
- sp's estimator is the CS-style version of (a)/(b): each group vs never-exposed clean controls, cohort-share weighted. So `did` on the constructed groups is an exact reference for estimates and SEs; fixest reproduces the regression form in the single-cohort case. The ring construction is recomputed independently in R (distance to nearest treated, bands, onsets) — counts and cell sets are asserted equal.

### harvest_did — attribution check

- Claimed paper: Abadie, Angrist, Frandsen & Pischke, "Harvesting Differences-in-Differences and Event-Study Evidence", NBER WP 34550 (Dec 2025). Verified to exist (nber.org/papers/w34550 and the MIT-hosted PDF; matches `paper.bib` key `abadie2025harvesting`). But its abstract/text: *"This paper surveys econometric innovations related to differences-in-differences estimators and event-study models…"*, described on its title page as a pre-publication draft chapter from *'Metrics Remastered: An Empiricist's Almanac* (Princeton University Press, 2026). Full-text search of the PDF: no "precision-weighted" / inverse-variance harvest estimator is defined; the only 2x2-aggregation mention is footnote 16 pointing to de Chaisemartin & D'Haultfœuille (2020). **The docstring's claim that `harvest_did` "implements the unified-estimation framework from" that paper was false** (and `registry.py` / `tests/test_harvest_did.py` further misattribute it to "Borusyak et al." / "Borusyak, Hull & Jaravel").
- What the function actually computes: CS ATT(g, g+e) cells with not-yet-treated controls and a universal base period g-1 — exactly `did::att_gt(control_group="notyettreated", base_period="universal")` — aggregated per horizon by a chosen weight and then across horizons by inverse variance.

### causal_impact — model check

- R CausalImpact (source read: `CausalImpact:::ConstructModel`): standardises data, local level (prior sd 0.01·sd(y)) + static regression with spike-and-slab prior (`expected.model.size`, `expected.r2`, `prior.df` constants), `bsts(..., seed = 1)` **hard-coded** — so the MCMC is deterministic and the R seed only moves the posterior-predictive draws (fixture: avg abs effect 3.64556539035117 identical across seeds 1–4; posterior sd 0.749–0.761).
- sp: OLS β on the pre-period, ρ = lag-1 correlation of residuals, σ_obs = sd(resid), σ_state = σ_obs·√(1-ρ²), Kalman filter for an AR(1) state + noise; no priors, no MCMC, no standardisation, no local level (the docstring said "local level / random walk"; the code is AR(1)). Same estimand label, different model. On the fixture: sp average effect 3.6928 (SE 0.285 after the fix) vs CausalImpact 3.6456 (posterior sd 0.761); true effect 4.0. Neither side is "wrong"; they are different estimators → **class 6**. A multi-seed T3 comparison is impossible (bsts seed fixed) and would be meaningless across models anyway.
- The deterministic chain sp runs is pinned to R: parameters to machine precision, KFAS::KFS one-step/forecast means and variances to 4.8e-15.

## 2. Defects found

1. **`spillover_did`: ring exposure timing ignored under staggered adoption.** First divergence: `did::aggte(simple)` on {ring-1 units with cohort = exposure onset} ∪ {clean} gave 1.2376 where sp gave 0.7912 (ring 2: 0.3966 vs 0.2524). Cause: every ring unit entered every treated cohort's (g, t) cell, so units near a late cohort were counted as exposed before they were, and units near an early cohort were differenced from an already-exposed base period. Fix: each ring unit's exposure onset = earliest cohort among treated units within the outer edge; ring cells are (ring, onset) groups vs clean controls from base onset−1; a warning (and `diagnostics["n_ring_changes"]`) when a unit moves to a closer ring after first exposure; `diagnostics["ring_onsets"]`. Single-cohort output is unchanged (bit-identical).
   Before → after on `did_synth_misc_spill_stag.csv` (truth: ring 1 ≈ 1.0–1.6 growing, ring 2 = 0.4): ring 1 **0.79123 → 1.23765**, ring 2 **0.25243 → 0.39661**. **Default output changed** (staggered designs only).
2. **`spillover_did`: cohort-share weight uncertainty omitted from SEs.** Direct SE 0.077918 vs did 0.091366 with the point estimate already exact → the missing term is `did`'s `wif()` (weights are estimated cohort shares). Added; it is identically zero with a single cohort. Before → after (staggered fixture): direct SE **0.077918 → 0.091366**, ring SEs 0.0702/0.0482 → 0.0879/0.0635. **Default output changed** (multi-cohort designs only).
3. **`harvest_did`: placebo cells used the treated cohort as its own control.** Control set was `never | cohort > max(t1, t2)`; for e ≤ −2, max(t1,t2) = g−1 < g, so cohort g was in both arms. `did` requires `G != g`. Fix: exclude cohort g. Before → after, cohort 5 at e=−4: **−0.3550 → −0.5104** (did −0.5104). **Default output changed** (pre-period horizons and the pre-trend test).
4. **`harvest_did`: aggregation assumed independent cells.** Event-study and aggregate variances were Σ w²·se², ignoring (i) shared control units across cohorts, (ii) the same treated units at every horizon (strongly positive covariance for the across-horizon aggregate), and (iii) estimated weights. Also the pre-trend Wald test used Σ (b/se)². Fix: unit-level influence functions per cell, event study = weighted IF sum (+ `wif` for `n_treated`), aggregate = weighted sum of horizon IFs, pre-trend Wald with the joint covariance (pinv + warning if singular). Monte Carlo (400 reps, known DGP, AR(1) errors, default weighting): MC sd of the aggregate 0.114; mean SE **0.059 before → 0.101 after** (residual gap from treating the data-dependent precision weights as fixed; documented). Default call on the fixture: aggregate SE **0.06207 → 0.10148**, estimate 1.79243 → 1.78531, pre-trend p 0.303 → 0.170. **Default output changed.**
5. **`harvest_did`: cell SE convention** changed from Welch (`ddof=1`) to `did`'s analytic influence-function SE (divisor n) — needed for a coherent joint covariance. A documented convention (both legitimate); ratio ≈ 1.012 on the fixture. Rolled into the ⚠️ entry for #4.
6. **`harvest_did`: false attribution** of the estimator to NBER WP 34550 (see §1). Docstring and summary header rewritten: building blocks attributed to `callaway2021difference`, the name explained as following the chapter title, the precision aggregation stated as StatsPAI's own. `registry.py` descriptions (two entries, "Borusyak et al.") need the same fix — forbidden file for me, see §7.
7. **`causal_impact`: SE of the average/cumulative effect ignored serial correlation of forecast errors.** `se_avg = sqrt(mean(se_t²))/sqrt(n_post)` treats post-period errors as independent, but they share the latent state: Cov(e_t, e_u) = ρ^{u−t} P_t. Found by deriving the model's own forecast covariance; fixed with the exact covariance and pinned by a reference-free identity (dense Gaussian conditioning of the whole series reproduces `se_total²` and the forecasts to 1e-10). Before → after on the fixture: SE **0.23480 → 0.28516**, `se_total` 7.0439 → 8.5547. Point estimate unchanged. **Default output changed** (whenever ρ ≠ 0).
8. **`causal_impact`: `n_seasons` silently ignored** (accepted, stored, never used). Now warns when > 1. Docstring said "local level / random walk" and "credible intervals"; the `__init__` said "Equivalent to Google's R CausalImpact package" — all corrected (AR(1) state, frequentist prediction intervals, explicitly *not* equivalent).

Also noted (not changed): R `did` 2.3.0 silently drops never-treated units when `gname` is an **integer** column (`data[g==0, g := Inf]` fails on integer → NA → "Dropped … observations that had missing data"). The generator stores gname as double; worth remembering for other did fixtures.

## 3. Proposed promotion records

```python
    "spillover_did": {
        "status": "bit-exact",
        "reference": "R did::att_gt(control_group='nevertreated') + did::aggte(type='simple') per group (direct / ring r, ring cohort = exposure onset); single cohort also fixest::feols(dbar ~ treat + ring1 + ring2, vcov='hetero', ssc(adj=FALSE))",
        "reference_versions": {"did": "2.3.0", "DRDID": "1.2.3", "fixest": "0.14.0"},
        "tolerance": (
            "Direct and ring effects, their SEs and every (group, onset cohort, period) cell at 1e-9 relative on a single-cohort and a staggered spatial panel; observed agreement 1e-14. The ring construction is recomputed independently in R from the coordinates."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_misc_parity.py"],
        "note": (
            "No package implements Butts's ring estimator; the regression step is pinned to the fixest form in Butts's own replication code and every design to did on the constructed groups. Promoted by removing two defects: under staggered adoption every ring unit entered every cohort's cell regardless of when it was exposed (ring effects 0.79 / 0.25 against did's 1.24 / 0.40), and the cohort-share weight term was missing from the standard errors (direct SE 0.078 vs 0.091). Single-cohort output is unchanged."
        ),
    },
    "harvest_did": {
        "status": "bit-exact",
        "reference": "R did::att_gt(control_group='notyettreated', base_period='universal') for every (cohort, horizon) cell; did::aggte(type='dynamic') for the event study under weighting='n_treated'",
        "reference_versions": {"did": "2.3.0", "DRDID": "1.2.3"},
        "tolerance": (
            "Every 2x2 cell (ATT and SE) and every event-study horizon (ATT and SE) at 1e-9 relative; observed agreement 1.7e-14. The inverse-variance aggregate over horizons (the default headline) has no reference and is checked by identity only."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_misc_parity.py"],
        "note": (
            "The estimator had been attributed to Abadie, Angrist, Frandsen & Pischke (NBER WP 34550, 2025), a survey chapter that defines no such estimator; its building blocks are Callaway-Sant'Anna cells. Promoted by removing defects: pre-period placebo cells counted the treated cohort among its own controls, and every aggregation (event study, headline aggregate, pre-trend Wald test) treated cells as independent although they share units -- headline SE 0.062 before, 0.101 after, Monte Carlo sd 0.114."
        ),
    },
```

`causal_impact`: **no promotion proposed.** Its only external agreement is the Kalman-filter step against KFAS (plus closed-form parameters); the estimator-level reference (CausalImpact) computes a different model. If the index ever grows a "component-only" tier, the evidence is `test_causal_impact_plugin_parameters`, `test_causal_impact_filter_matches_kfas` (KFAS 1.6.0, 1e-9, observed 4.8e-15) and the identity test.

## 4. Proposed CHANGELOG / MIGRATION

CHANGELOG — ⚠️ Correctness:

- ⚠️ `sp.spillover_did`: under staggered adoption each spillover ring is now measured from its own exposure onset (the first period a treated unit lies within the outer ring edge), as in Butts's staggered applications; previously every ring unit entered every treated cohort's comparison regardless of when it was exposed, which biased the ring effects toward zero. Standard errors now include the influence function of the cohort-share weights (zero with one cohort). Single-cohort designs are unchanged. Now pinned to R `did` and `fixest` at 1e-9.
- ⚠️ `sp.harvest_did`: pre-period placebo cells no longer count the treated cohort among its own controls; event-study, aggregate and pre-trend inference now use the joint covariance of the cells (unit-level influence functions) instead of assuming independence, which had understated the headline SE by about 40%; cell SEs use `did`'s analytic convention (divisor n instead of n−1). With `weighting="n_treated"` cells and event study reproduce `did::att_gt(control_group="notyettreated", base_period="universal")` / `did::aggte(type="dynamic")`.
- ⚠️ `sp.causal_impact`: the standard error of the average / cumulative effect now accounts for the serial correlation of the post-period forecast errors (they share the latent AR(1) state); previously they were treated as independent, understating the SE whenever the fitted ρ > 0. Point estimates unchanged.

CHANGELOG — Fixed:

- `sp.harvest_did` no longer claims to implement Abadie, Angrist, Frandsen & Pischke (2025), a survey chapter that defines no such estimator; the docstring now describes the Callaway–Sant'Anna building blocks and the StatsPAI-specific precision aggregation.
- `sp.causal_impact`: `n_seasons` was accepted and silently ignored; it now warns. Docstrings no longer describe the model as a local-level / Bayesian model "equivalent to" R `CausalImpact` — it is a frequentist regression + AR(1) state-space model.

CHANGELOG — Added:

- `tests/reference_parity/test_did_synth_misc_parity.py`: `spillover_did` and `harvest_did` against R `did` / `fixest`; `causal_impact`'s filter step against `KFAS`.

MIGRATION rows:

| function | what changed | before → after (fixture) | how to get the old number |
| --- | --- | --- | --- |
| `sp.spillover_did` (staggered only) | ring effects by exposure onset; SE adds weight IF | ring 1 0.7912 → 1.2376; direct SE 0.0779 → 0.0914 | not reachable (old value was not a documented estimand) |
| `sp.harvest_did` | own-cohort exclusion in placebos; joint-covariance inference; IF cell SE | headline SE 0.0621 → 0.1015; cohort-5 e=−4 cell −0.3550 → −0.5104 | not reachable (old SEs assumed independence that does not hold) |
| `sp.causal_impact` | SE uses correlated forecast errors | SE 0.2348 → 0.2852 | `sqrt(mean(detail.predicted_se[post]**2)/n_post)` recomputes the old independence formula |

## 5. Not closed / caveats

- `causal_impact` vs R `CausalImpact`: class 6 (different model; see §1). Also `bsts` seed is hard-coded to 1 inside CausalImpact, so even a T3 multi-seed study of CausalImpact itself is not available without calling `bsts` directly. sp's SE still ignores parameter uncertainty (β, ρ, variances) — documented in the docstring; that is why it is well below CausalImpact's posterior sd (0.29 vs 0.76) on the fixture.
- `harvest_did` default `weighting="precision"` headline: no reference; weights treated as fixed (400-rep MC on a known DGP: mean SE / MC sd = 0.89 for the default headline, 0.91 with `n_treated` weights; for comparison the event-study horizon e=2, which equals `did`'s own analytic SE exactly, is at 0.93 on the same DGP, so most of the shortfall is shared with `did`'s analytic SE at n=150).
- `spillover_did`: units that move to a closer ring after first exposure are counted in their final ring from first exposure (warned, counted in diagnostics). Butts's time-varying ring membership would need a unit-period exposure mapping; not implemented. No Stata side (no Stata implementation exists; `ssc`/Stata not checked beyond that because the estimator has no Stata command).
- Stata: no references attempted for this cluster (no Stata ports of the ring estimator or CausalImpact; `csdid` is a bridge for `did`, which is already the canonical reference).

## 6. .gitignore

Nothing needed (no Stata ado dir used; the Butts repos and the NBER PDF were cloned/downloaded into the session scratchpad only).

## 7. For the lead (forbidden files I did not touch)

- `src/statspai/registry.py`: `harvest_did` description (line ~7972) says "Borusyak et al. MIT/NBER 34550" — wrong on two counts; the `causal_impact` description says "Bayesian structural time series" — it is a frequentist regression + AR(1) state-space model.
- `src/statspai/_agent_cards_extra.py` `harvest_did` card: assumption text "cross-horizon covariance ignored" is now stale.
- `tests/test_harvest_did.py` module docstring misattributes to "Borusyak, Hull & Jaravel".
- `docs/parity.md` / index: `spillover_did` currently listed as analytical-only with the note "no cross-package reference".
