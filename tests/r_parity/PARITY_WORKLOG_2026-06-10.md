# Track A parity hardening worklog — 2026-06-10

Scope: harness-wide hardening pass over the cross-language parity harness
(`tests/r_parity` + `tests/stata_parity`), expanding Track A from 56 to
64 modules and re-certifying both reproducibility legs from scratch.
No JOSS-paper artefacts (`paper.md` / `paper.bib`) were touched.

## 1. Baseline re-certification (before any change)

- `compare.py` regenerated all four tables byte-identically to the
  committed artefacts (no drift between JSONs and rendered tables).
- R-side `verify_reproduce.py`: 53/53 discovered modules reproduce with
  worst rel diff 0.00e+00 under R 4.5.2 / vecLib.
- Stata-side `verify_reproduce_stata.py`: 53/53 modules reproduce with
  worst rel diff 0.00e+00 under Stata 18 MP.
- `tests/reference_parity/`, `tests/external_parity/` (65 passed,
  1 optional-dep skip), `test_parity_runtime.py`,
  `test_ci_exact_parity_contract.py`, `test_orig_parity_harness_contract.py`,
  `test_parity_gap_boundaries.py`: all green.
- Fixed two stale inventory assertions in
  `tests/test_parity_harness_contract.py` left over from the
  `50_xtabond` R-reference materialization (R-side gap set is now empty;
  machine tier count 49 -> 50).

## 2. New Track A modules 57–64 (py + R + Stata, all three sides)

| # | Module | StatsPAI | R reference | Stata reference | Tier |
|---|---|---|---|---|---|
| 57 | Binary logit | `sp.logit` | `stats::glm(binomial("logit"))`, epsilon=1e-12 | `logit` | machine |
| 58 | Poisson ML | `sp.poisson` | `stats::glm(poisson())`, epsilon=1e-12 | `poisson` | machine |
| 59 | LIML k-class | `sp.liml` | `ivmodel::LIML` | `ivregress liml, small` | machine |
| 60 | SUR one-step FGLS | `sp.sureg` | `systemfit(SUR, noDfCor, maxiter=1)` | `sureg` | machine |
| 61 | Beta regression | `sp.betareg` | `betareg(link.phi="log")` | `betareg, nrtolerance(1e-13)` | machine |
| 62 | Truncated regression | `sp.truncreg` | `truncreg(method="NR")` | `truncreg, ll(0)` | machine |
| 63 | Zero-inflated Poisson | `sp.zip_model` | `pscl::zeroinfl(poisson)`, reltol=1e-14 | `zip, inflate(...)` | machine |
| 64 | Zero-inflated NB | `sp.zinb` | `pscl::zeroinfl(negbin)`, reltol=1e-14 | `zinb, inflate(...) nrtolerance(1e-13)` | iterative (1e-5 budget) |

Convention findings nailed down while aligning the three sides:

- **59_liml**: sp.liml and `ivmodel::LIML` agree at 1e-15 including SEs.
  Stata `ivregress liml` defaults to the RSS/n error-variance divisor;
  `small` puts it on the shared RSS/(n-k) convention (the observed 1.0e-3
  SE gap collapses to ~3e-15).
- **60_sureg**: sp.sureg's one-step FGLS with Sigma = E'E/n matches the
  Stata `sureg` default exactly; `systemfit` needs
  `methodResidCov='noDfCor'` (its default `geomean` corresponds to
  Stata's `dfk` option instead). All 6 coefficients + SEs at ~1e-15.
- **61_betareg**: point estimates are machine-level on all three sides.
  `betareg` reports expected-(Fisher-)information SEs while sp/Stata
  report observed-information SEs — a documented convention gap of
  <=0.7% on this fixture (numDeriv observed-info SEs at the betareg
  optimum reproduce the sp/Stata SEs). Registered `rel_se=1e-2` with an
  inline justification; py<->Stata SEs agree at ~1e-6.
- **62_truncreg**: `truncreg`'s default BFGS stops ~2.5e-5 short of the
  optimum; `method="NR"` converges to the same likelihood optimum as
  sp/Stata (logLik gain ~5e-8), bringing the worst gap to ~1e-7.
  Sigma rows compare on the natural scale (sp delta-maps exp(ln_sigma)).
- **63/64 ZIP/ZINB**: Stata's default `nrtolerance(1e-5)` leaves ~5e-6
  coefficient slack on the flat ZINB likelihood; `nrtolerance(1e-13)`
  closes it. The ZINB likelihood is genuinely flat near the optimum
  (R EM vs BFGS refinements move coefficients ~3e-7 at logLik identical
  to 1e-10), so 64 registers a 1e-5 point budget (iterative tier)
  instead of forcing a brittle machine claim.

Strictness tiers after expansion: **57 machine / 5 iterative /
1 moderate / 1 methodological (T4)** across 64 rendered modules;
61 of 64 have a Stata reference (13/18/19 skip reasons unchanged and
re-verified: Stata 18 has no `cate`; `allsynth` rejects the Basque
fixture and follows a different estimand; `fect_stata` does not recover
the R gsynth convention).

## 3. Harness hardening

- `verify_reproduce.py::discover_modules` no longer requires a
  `data/<module>.csv`: 10_honest_did / 21_honest_relmags / 23_evalue
  embed their fixtures in-script and are now part of the automated
  R-reproducibility gate (53 -> 64 verified modules).
- `_gen_renv_lock.R::REFERENCE_PKGS` was missing the canonical packages
  for modules 22/23/28/31/32/38/41/43/44/53 (sensemakr, EValue, sfaR,
  ddecompose, dineq, DRDID, censReg, sampleSelection, nnet,
  clubSandwich) — plus the new 57–64 packages (ivmodel, systemfit,
  betareg, truncreg, pscl). `renv.lock` regenerated: 285 packages
  (43 reference + transitive closure).
- `R_ENVIRONMENT.md` reference-package table extended accordingly
  (betareg 3.2.4, truncreg 0.2.5, ivmodel 1.9.1, systemfit 1.1.30,
  pscl 1.5.9, and the previously missing module packages).

## 4. Post-expansion certification

- R-side `verify_reproduce.py`: all 64 modules reproduce (0 drift).
- Stata-side `verify_reproduce_stata.py`: all 61 modules reproduce
  (0 drift), including the 8 new ones.
- `test_parity_harness_contract.py` inventory/tier assertions updated
  (machine 57, iterative 5; py == R == TOLERANCES == HEADLINE over 64
  modules) and green.
- `TIER_A_FIXTURE_LOCK.json` refreshed via
  `python scripts/tier_a_fixture_lock.py --write` after the expansion.
