# Design Rationale & Frequently-Raised Questions

This page answers the questions reviewers most often raise about a package
of StatsPAI's scope, with pointers to the concrete evidence in the
repository. It complements the [JOSS reviewer guide](joss_reviewer_guide.md)
(how to install and smoke-test) and the
[JOSS validation dossier](joss_validation_dossier.md) (the validation
evidence itself). Every claim below links to a file you can open.

---

## Scope & quality

### With 1,000+ functions across 81 submodules, how can a reviewer trust the breadth?

StatsPAI does **not** ask reviewers to take 1,020 functions on faith. Three
mechanisms make the breadth auditable rather than assertable:

1. **Validation tiering is explicit and machine-readable.** Parity-backed
   functions expose a `validation_status` so a human or agent can tell
   *certified numerical evidence* apart from *API-stable breadth*. Nothing
   is silently presented as validated when it is not.
2. **Independent validation tracks, not self-comparison.** The numerical
   core is pinned against external references:
   - **124 reference-parity checks** (`tests/reference_parity/`) against R,
   - **54 external-parity checks** (`tests/external_parity/`) against the
     canonical upstream Python implementations (e.g. DoubleML),
   - R/Stata cross-language parity (`tests/r_parity/`, `tests/stata_parity/`),
   - Monte-Carlo coverage checks (`tests/coverage_monte_carlo/`),
   - original-paper replays (`tests/orig_parity/`).
   The archived full-suite report records **5,200 passed** on Python 3.9
   (`test_results_full_suite.md`).
3. **Coverage is honest and rising.** The CI line-coverage gate was lifted
   from a 15% placeholder to **60%** against a measured **70.5%** baseline
   (historical full-suite coverage XML snapshot). The current CI gate uses the
   `coverage.xml` generated inside the same job, while `.coverage_campaign/`
   tracks the dedicated campaign driving the six core estimator modules
   (`did iv rd synth dml panel`) to **≥95%** line coverage under the full
   suite — test-only, with real numerical assertions, no mocking of numerical
   paths.

The honest framing — also stated in the validation dossier — is that
*certified* methods carry parity or analytic evidence, while *breadth*
methods are API-stable and documented; the `validation_status` field is the
contract that keeps the two from being conflated.

### Isn't a package this large just thin wrappers around existing tools?

No. A grep of the source shows exactly one wrapper boundary
(`src/statspai/fixest/wrapper.py`, an opt-in pyfixest bridge); the
estimators are first-party implementations. StatsPAI *depends on* the
scientific-Python stack (NumPy, SciPy, pandas, statsmodels, scikit-learn,
linearmodels) and implements the causal/econometric layer on top of it
rather than re-exporting another package's estimators.

---

## Double Machine Learning (`sp.dml`)

### Given that DoubleML already exists, why does `sp.dml` exist? What is the increment?

`sp.dml` is **not** a competing DML engine — and the documentation says so
plainly. Its value is integration, not reimplementation of the theory:

- **One import, one result object.** The same `sp.dml(model=...)` dispatcher
  sits behind `import statspai as sp` next to DiD, IV, RD, synthetic
  control, meta-learners, etc., and returns the shared `CausalResult`
  (`.summary()`, `.to_latex()`, `.to_word()`, `.cite()`). A user moving
  between designs does not switch libraries or result conventions.
- **Agent-native surface.** Every estimator, `sp.dml` included, is
  discoverable via `list_functions()` / `describe_function()` /
  `function_schema()`, so the same entry point serves humans and agents.
- **Cross-ecosystem alignment.** `sp.dml` is pinned against *both* DoubleML
  ecosystems (R and Python), so the numerical claim is auditable from
  either side.

The increment is a unified, agent-native, R/Stata-aligned interface over a
correctly-implemented orthogonal-score core — not a claim to improve on the
DoubleML estimator itself.

### Are the DML scores correct (Neyman-orthogonal) and is the cross-fitting faithful?

Yes, and this is verified numerically rather than asserted. Under identical
scikit-learn learners and the same cross-fit fold partition with a fixed
seed:

| Model | Agreement with `doubleml-for-py` |
| --- | --- |
| **PLR** (partially linear) | machine precision — \|Δ coef\| ≈ 1.1×10⁻¹⁶, \|Δ SE\| ≈ 1.4×10⁻¹⁷ |
| **PLIV** (partially linear IV) | machine precision under shared learners/folds |
| **IRM** (AIPW ATE) | within ≈ 0.10 SE; residual traced to AIPW score construction (not trimming, not IPW normalization) |
| **IIVM** (interactive IV / LATE) | analogous orthogonal score; pinned alongside IRM |

Machine-precision PLR/PLIV agreement is only possible if both
implementations evaluate the *same* Neyman-orthogonal score on the *same*
cross-fit partition. The full numbers, software versions, and an honest
discussion of where and why `sp.dml` diverges are in
[`docs/guides/sp_dml_vs_doubleml.md`](guides/sp_dml_vs_doubleml.md); the
checks themselves are
[`tests/external_parity/test_dml_python_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/external_parity/test_dml_python_parity.py)
(Python) and `tests/reference_parity/test_dml_parity.py` (R).

To reproduce:

```bash
python -m pip install -e ".[dev,parity]"   # parity extra adds doubleml-for-py
python -m pytest tests/external_parity/test_dml_python_parity.py -v
```

---

## Rigorous Lasso (`sp.rlasso`, the `hdm` port)

### Given that `hdm` already exists, why does `sp.rlasso` exist? How faithful is the port?

`sp.rlasso` is **not** a re-derivation of rigorous Lasso — it is a deliberate,
bit-level port of the R `hdm` package (Chernozhukov, Hansen & Spindler, *The R
Journal* 8(2), 2016) into the same agent-native, one-import workspace as the
rest of StatsPAI. The increment is the *interface and integration*, while the
numerics are held to `hdm` exactly:

- **One import, one result surface.** `sp.rlasso`, `sp.rlasso_effect`,
  `sp.rlasso_effects`, `sp.rlasso_iv`, and `sp.rlassologit` sit behind `import
  statspai as sp` and feed straight into `sp.dml` as nuisance learners
  (`ml_g='rlasso'`), so the rigorous-penalty estimator and the DML assembly
  share one validated implementation.
- **Agent-native.** All of them are discoverable via `list_functions()` /
  `describe_function()` / `function_schema()`, the same as every other
  estimator.
- **Faithful, not approximate.** An earlier from-memory reconstruction of
  optimal-IV selection was ~17× off `hdm` on the eminent-domain application
  because it used the *asymptotic* penalty and selected only instruments; that
  path is deprecated. `sp.rlasso` ports the rigorous penalty, the
  heteroskedastic loadings, the LassoShooting objective, and post-Lasso exactly.

### Is the rigorous-Lasso port numerically correct?

Yes, and it is pinned, not asserted. Because `hdm`'s penalty is data-driven
(no cross-validation), its output is deterministic, so the fixtures are a hard
contract (`hdm` 0.3.2 / R 4.5.2 / `glmnet` 4.1.10) — **29 parity tests, all
passing**:

| `hdm` surface | Agreement with `hdm` |
| --- | --- |
| `rlasso` coefficients / `λ₀` / loadings / residuals | ~1e-13; **selected support exact** |
| `rlasso_effect` / `rlasso_effects` (α, SE) | ~1e-14 |
| `rlasso_iv` eminent domain, `select_Z` (BCH 2012) | `coef 0.2274`, `SE 0.2466` (~1e-9) |
| `rlassologit` selected support / `post` refit | support exact; `post` ~1e-9 |
| `sp.dml(ml_g='rlasso', ml_m='rlasso')` end to end | matches R DoubleML `θ̂=1.44867`, `SE=0.04502` (machine precision) |
| `hdm` vignette applications | Growth, AJR, and cps2012 coefficients pinned to `hdm` |

The full numbers and the `hdm` ↔ StatsPAI function map are in
[`docs/guides/rigorous_lasso_hdm.md`](guides/rigorous_lasso_hdm.md); the
evidence dossier section is *Rigorous Lasso (`hdm`) Parity* in
[`docs/joss_validation_dossier.md`](joss_validation_dossier.md).

To reproduce (no R needed — the `hdm` reference is committed as JSON):

```bash
python -m pytest tests/reference_parity/test_rlasso_parity.py -v
python -m pytest tests/reference_parity/test_rlassologit_parity.py -v
python -m pytest tests/reference_parity/test_rlasso_vignette_parity.py -v
```

---

## Installation & dependencies

### Does the package install cleanly, and are heavy dependencies optional?

Yes. The core runtime declares only the scientific-Python stack; `torch`,
`jax`, and `pymc` live in optional extras (`neural`/`deepiv`, `performance`,
`bayes`) and are **lazily imported**. CI proves this with a bare-install
smoke step: it builds the wheel, installs it into a fresh virtual
environment with **no extras**, and runs core estimators (IV on the Card
data, staggered DiD on the MPDTA data) — so a regression that eager-imports
an optional dependency fails CI rather than a downstream user's
`pip install statspai` (see the *bare install — lazy-import guard* step in
`.github/workflows/ci-cd.yml`).

---

## Reproducibility

### Can a reviewer reproduce the parity claims without installing R and Stata?

Yes. The R/Stata reference outputs are captured once and committed as
fixtures (JSON), so the reference-parity suite runs against the cached
numbers with only a Python toolchain:

```bash
python -m pytest tests/reference_parity/ -q --no-cov
```

R or Stata is needed only to *regenerate* a fixture (when a DGP changes),
not to *run* the parity check. Each fixture records the exact upstream
software version it was generated with, so the comparison is pinned and
documented rather than live and drifting.

### Why does the version under review differ from the latest release?

Development continued during review. The reviewed numerical evidence (parity
tables, dossier numbers) is pinned to committed fixtures and tests, so it is
reproducible at the reviewed commit regardless of later releases; the
maintainer coordinates the exact archival version with the editor.

---

## Documentation

Full documentation, including the API reference generated directly from
NumPy-style docstrings, is published at
<https://brycewang-stanford.github.io/StatsPAI/> and builds under
`mkdocs build --strict` as a CI gate.
