# JOSS Validation Dossier

This dossier collects reviewer-facing evidence for StatsPAI's readiness as
research software. It is intentionally factual and reproducible.

## Project Status

- Repository: <https://github.com/brycewang-stanford/StatsPAI>
- Package archive: <https://doi.org/10.5281/zenodo.19933900>
- PyPI: <https://pypi.org/project/StatsPAI/>
- License: MIT, with a plain-text `LICENSE` file in the repository.
- Current release at the time of this dossier: `1.16.0`, released on
  2026-05-29.
- Public GitHub repository creation date: 2025-07-26.
- Public repository activity signals as of 2026-06-01: 212 stars, 39 forks,
  23 GitHub releases, and 1 public external user issue in addition to
  maintainer-created issue/PR activity.

## Software Scope

StatsPAI exposes a unified Python interface for causal inference and applied
econometrics. As of release `1.16.0`, the registry reports 1,020 public
functions across 81 submodules:

```bash
python scripts/registry_stats.py --check
```

The registry and schema layer are part of the public surface. They support
programmatic discovery through `sp.list_functions()`, `sp.describe_function()`,
and `sp.function_schema()`.

## Validation Assets

The repository includes several independent validation tracks:

- Unit and integration tests across the main estimator families.
- R parity modules under `tests/r_parity/`.
- Stata parity modules under `tests/stata_parity/`.
- Reference-parity checks under `tests/reference_parity/`.
- Original-paper replay fixtures under `tests/orig_parity/`.
- Monte Carlo coverage checks under `tests/coverage_monte_carlo/`.
- Snapshot tests for publication-table output under `tests/output_snapshots/`.
- Citation and bibliography audits under `tools/`.
- Reviewer-facing offline examples under `examples/`.

The archived local full-suite report records:

```text
5200 passed, 98 skipped, 13 deselected, 1 xfailed, 2 xpassed
```

on Python 3.9.6 for the default local suite as of 2026-05-17. The exact report
is stored in `test_results_full_suite.md`.

## Parity And Replication Anchors

StatsPAI includes validation fixtures for common teaching and replication
benchmarks, including:

- Card-style returns-to-schooling IV estimates.
- LaLonde / Dehejia-Wahba job-training benchmarks.
- Lee-style close-election regression discontinuity.
- Callaway-Sant'Anna difference-in-differences examples.
- California Proposition 99 synthetic-control examples.
- **Hernán & Robins, *Causal Inference: What If* (NHEFS).** The first
  public-health / epidemiology parity anchor, on **real** public-domain
  data (`sp.datasets.nhefs()`). StatsPAI reproduces the textbook's
  published g-methods estimates for the effect of quitting smoking on
  10-year weight change and mortality — IP weighting (Ch12, 3.47 vs book
  3.4), standardization / parametric g-formula (Ch13, 3.46 vs 3.5),
  g-estimation of a structural nested model (Ch14, 3.46 vs 3.4), outcome
  regression with effect modification (Ch15, coefficients matching the
  book to four decimals), IP-weighted survival (Ch17), and an E-value
  sensitivity analysis. Each statistic carries a same-bytes R gold
  reference (base R / `survival` / `EValue`); StatsPAI matches R to
  machine precision (≤1e-9) on the closed-form quantities and the
  published book to ~2% on the iterative ones. Three g-methods agreeing
  with each other and the book (~3.4–3.5 kg) is the canonical Part-II
  triangulation. Paired scripts: `tests/orig_parity/06–11_nhefs_*.{py,R}`;
  rollup `tests/orig_parity/results/parity_table_orig.md`; pinned tests
  `tests/external_parity/test_whatif_nhefs.py`; primary-source anchors in
  `tests/external_parity/PUBLISHED_REFERENCE_VALUES.md`; worked-example
  walkthrough in `docs/guides/whatif_nhefs.md`.

### Findings surfaced by the *What If* reproduction

Reproducing a published reference end-to-end is the most effective audit
of an implementation. This exercise surfaced two items, documented here
rather than hidden:

1. **Modelling-convention differences (expected, not defects).**
   `sp.g_computation`'s `covariates=` API takes a flat list of columns
   and so fits an *additive* outcome model; it cannot express the book's
   `qsmk:smokeintensity` effect-modification term. Its standardized ATE
   (3.46) matches an additive base-R/Python standardization gold to 12
   significant figures and rounds to the book's 3.5; the book's *exact*
   interaction-model standardization (3.52) is reproduced directly. The
   IP-weighting (Hájek vs stabilized weights) and SNMM (additive
   encoding) differences are of the same documented kind.
2. **A minor correctness gap in `sp.evalue` (CI handling) — fixed.**
   When a confidence interval already crosses the null (RR = 1), the
   E-value for the confidence limit should be exactly 1 (VanderWeele &
   Ding 2017; the R `EValue` package returns 1). `sp.evalue` previously
   computed the E-value of the far CI limit (e.g. RR = 0.90, CI
   (0.79, 1.22) returned `evalue_ci` 1.74 rather than 1.0). The
   point-estimate E-value was always correct and matches the closed form
   / `EValue` to 1e-3; only the CI-limit branch lacked a null-crossing
   guard. This was fixed by clamping the relevant CI limit to the null
   when the interval contains it, with regression tests in
   `tests/test_evalue.py`; the existing R/Stata E-value parity values are
   unchanged (their cases do not cross the null on the far side).

Known convention differences are documented in parity reports rather than
hidden. For example, bandwidth selectors, regularisation constants, small-sample
standard-error conventions, and fold-split randomness are recorded in the
R-parity report where they affect exact numerical matching.

## Double Machine Learning Parity

`sp.dml` is StatsPAI's port of the Double/Debiased Machine Learning framework
(Chernozhukov et al., 2018). Because the canonical reference implementations
are the `DoubleML` packages for R and Python (Bach, Chernozhukov, Kurz,
Spindler & Klaassen), `sp.dml` is pinned against **both** so the numerical
claim is auditable from either ecosystem.

The fixture is a fixed seed-42 DGP (`n=1000`, `p=10`, true effect `θ=0.5`) at
`tests/reference_parity/_fixtures/dml_data.csv`. All three engines consume the
same CSV. On the Python side, `sp.dml` and `doubleml-for-py` are given
**identical** scikit-learn nuisance learners (`LassoCV(cv=5)` for regression,
`LogisticRegressionCV(cv=5)` for the binary propensity) and the same fold
partition under a fixed seed, so any divergence reflects a genuine
implementation difference rather than learner choice or split noise.

All four DoubleML model classes are pinned against `doubleml-for-py`.
The non-instrumented models (PLR, IRM) use `dml_data.csv`; the
instrumented models (PLIV, IIVM) use the companion `dml_iv_data.csv`
(n=2000, continuous instrument `z_c`, binary instrument `z_b`).

| Model | `sp.dml` (StatsPAI 1.16.1) | `doubleml-for-py` 0.11.3 | `DoubleML` R 1.0.2 (cv.glmnet) |
| --- | --- | --- | --- |
| **PLR** (continuous D) | +0.559022 ± 0.033103 | +0.559022 ± 0.033103 | +0.536759 ± 0.033498 |
| **IRM** (binary D, AIPW ATE) | −0.019107 ± 0.076561 | −0.026658 ± 0.074206 | +0.006640 ± 0.074434 |
| **PLIV** (continuous D, instrument) | +0.511701 ± 0.019453 | +0.511701 ± 0.019453 | — (Python-side pin) |
| **IIVM** (binary D, instrument, LATE) | +0.549467 ± 0.092426 | +0.561773 ± 0.091915 | — (Python-side pin) |

- **PLR matches `doubleml-for-py` to machine precision.** Under shared learners
  and folds the point estimate and standard error agree to within one float64
  unit in the last place: |Δ coefficient| = 1.1 × 10⁻¹⁶ and |Δ standard error|
  = 1.4 × 10⁻¹⁷. This is exact numerical equivalence, not a loose tolerance —
  both implementations evaluate the same Neyman-orthogonal score on the same
  cross-fit partition. The corresponding deviation from the R reference is
  ~4.1% on the coefficient, attributable to `cv.glmnet`'s penalty path
  differing fractionally from scikit-learn's `LassoCV`; the R fixture is pinned
  to within 7% relative.
- **IRM agrees within one-tenth of a standard error.** `sp.dml` and
  `doubleml-for-py` differ by 0.0076 on the ATE (≈ 0.10 SE on this fixture);
  all three implementations are statistically indistinguishable from zero, the
  truth for this DGP. The residual difference comes from internal AIPW
  score-construction details — it is verified *not* to be driven by propensity
  trimming (matching the clip thresholds leaves it unchanged) nor by IPW
  normalization (toggling `normalize_ipw` leaves it unchanged). The external
  parity test pins this at < 0.05 absolute.

Both directions are exercised by committed tests, not just asserted in prose:

```bash
python -m pip install -e ".[dev,parity]"   # the parity extra adds doubleml-for-py
python -m pytest tests/external_parity/test_dml_python_parity.py -v   # sp.dml vs doubleml-for-py (machine precision)
python -m pytest tests/reference_parity/test_dml_parity.py -v          # sp.dml vs DoubleML R (needs local R + DoubleML)
```

The Python-side check runs whenever `doubleml-for-py` is installed (via the
`parity` extra) and skips cleanly otherwise; the R-side check additionally
requires a local R installation with `DoubleML` 1.0.2 + `mlr3learners` 0.14.0.
Environment of record for the numbers above: StatsPAI 1.16.1, `doubleml-for-py`
0.11.3, scikit-learn 1.7.2, and `DoubleML` R 1.0.2 on R 4.5.2 with `cv_glmnet`.
The API mapping between `sp.dml(model=...)` and the DoubleML classes, plus the
full divergence discussion, is in `docs/guides/sp_dml_vs_doubleml.md`.

## Research Use

At submission time, StatsPAI is being used in working-paper workflows connected
to the Stanford Rural Education Action Program and related empirical policy
evaluation work. No peer-reviewed research article using StatsPAI has yet been
published. The current impact claim is therefore based on credible near-term
research use, reproducible validation materials, public package distribution,
and reviewer-verifiable examples rather than published downstream citations.

## Public Distribution And Community Signals

StatsPAI is publicly distributed on PyPI and archived on Zenodo. The GitHub
repository has public stars, forks, issue templates, a dedicated support
discussion channel, contribution instructions, support instructions, release
notes, and CI status checks. These are treated as community-readiness and
public-interest signals, not as evidence of independent scholarly adoption.

The public fork list is available through GitHub at
<https://github.com/brycewang-stanford/StatsPAI/forks>. As of 2026-05-29, the
GitHub API reported 37 forks, all owned by normal GitHub `User` accounts. The
project does not infer downstream research use from those forks unless a user
opens an issue, pull request, citation, or reproducible report that documents
such use.

## Commercial Downstream Disclosure

StatsPAI Inc. is the legal entity associated with the project. CoPaper.AI is a
commercial downstream product that may call the MIT-licensed StatsPAI package.
The StatsPAI package itself is permanently open source under the MIT license.
This is an open-core / commercial-downstream arrangement: the research software
submitted to JOSS remains open, while commercial products can build on it under
the same license terms available to all users.

## Reproducible Checks

From a repository checkout:

```bash
python -m pip install -e ".[dev,plotting]"
python -m pytest tests/test_ols.py tests/test_did.py tests/test_registry.py -q --no-cov
python scripts/registry_stats.py --check
python scripts/schema_quality.py
python tools/audit_bib_duplicates.py --strict
python tools/audit_bib_coverage.py --strict-dangling --hide-orphans
python -m build
python -m twine check dist/*
```

For a shorter package-level check, use the reviewer guide in
`docs/joss_reviewer_guide.md`.
