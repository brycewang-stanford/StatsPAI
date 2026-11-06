# StatsPAI against the rOpenSci statistical software standards

Audit date 2026-11-05. StatsPAI 1.42.1 (main at `831f154e1`).

## What the yardstick is

rOpenSci runs peer review for statistical software. Its "stats dev guide" lists the standards that reviewers check. There is one general set and one set per category. This audit uses four sets. They are General, Regression and Supervised Learning, Bayesian and Monte Carlo, and Time Series.

- Page fetched: https://stats-devguide.ropensci.org/standards.html. The page title reads "Standards: Version 0.2.0". The page shows no date.
- Standard text was read from the book source, github.com/ropensci/statistical-software-review-book, files `standards/general.Rmd`, `regression.Rmd`, `bayesian.Rmd` and `time-series.Rmd`. Source commit `13acd72`, committed 2026-10-20. The repo's `_variables.yml` gives version 0.2.0, matching the page.
- Every ID below comes from those files. There are 227 IDs in the four sets (68 G, 48 RE, 56 BS, 55 TS). Sub-items such as G2.4a are counted as their own rows.

## Method

1. Each standard was paraphrased. R mechanics were mapped to Python. roxygen2 maps to NumPy docstrings plus the registry schema. `NA` maps to `NaN` / `pd.NA`. `print` / `summary` / `plot` S3 methods map to `__repr__` / `.summary()` / `.plot()`. `testthat` maps to pytest. An R-only idiom with no Python analogue is marked not applicable, with the reason.
2. Evidence came from reading the repo and from running code in `.venv`. Probe scripts fed edge-case inputs to a sample of estimators across families: `sp.regress`, `sp.feols`, `sp.logit`, `sp.glm`, `sp.ologit`, `sp.callaway_santanna`, `sp.rdrobust`, `sp.synth`, `sp.dml`, `sp.causal_forest`, `sp.arima`, `sp.ets`, `sp.var`, `sp.corrgram` and `sp.bayes_regress` (the `statspai.mcmc` Gibbs / Metropolis sampler). The inputs were missing values, `inf`, zero rows, all-NaN columns, constant columns, perfect collinearity, strings, `Decimal` and list columns, int64 near 2^62, integer overflow in formula terms, bad enum values, seeds, and convergence limits.
3. The package-wide counts come from the repo's own audit scripts. These are `scripts/help_coverage.py`, `scripts/seed_inventory.py`, `scripts/error_taxonomy_audit.py`, `sp.parity_summary()` and `sp.function_schema()` over all 1,680 registered functions.
4. Status values are met, partial, gap, n/a, and not checked. "Met" means I saw or ran the evidence. "Partial" means some sampled functions meet it and others do not, or the mechanism exists but is not uniform. A probe of about 15 functions cannot certify 1,680. Treat "met" as "met in the sample and by the shared mechanism", not as a census.
5. Several TS standards are "either / or" choices (for example TS4.0a or TS4.0b). When one alternative is met, the others are marked n/a.

## Summary

| Category | IDs | met | partial | gap | n/a | not checked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General (G) | 68 | 26 | 35 | 6 | 1 | 0 |
| Regression (RE) | 48 | 21 | 19 | 6 | 2 | 0 |
| Bayesian and Monte Carlo (BS) | 56 | 29 | 14 | 9 | 3 | 1 |
| Time Series (TS) | 55 | 21 | 17 | 7 | 10 | 0 |
| **Total** | **227** | **97** | **85** | **28** | **16** | **1** |

The short reading follows. Correctness testing (G5.4 to G5.6) is the strongest area. It goes beyond what the standards ask, with same-byte R / Stata parity, NIST StRD, exact-posterior grids and B=1000 coverage runs. The weak area is input hygiene at the edges. Missing and non-finite values are handled one way in one estimator and another way in the next. A few paths are silent, and two of them give wrong numbers with no warning (`sp.synth` with a missing treated-unit value, `sp.ologit` with string outcome levels).

## Probe results that drive the gaps

These are the findings that the tables cite as "probe". All were run on 1.42.1.

| # | Call | Observed |
| --- | --- | --- |
| P1 | `sp.synth`, one NaN in the treated unit's pre-period outcome | No warning. `pre_treatment_mspe` is NaN, the solver loss is NaN, donor weights fall back to equal 1/11, `converged=True`, and the estimate moves from 2.009 to 1.856. |
| P2 | `sp.synth`, one NaN in a donor's outcome, or a missing donor row | No warning. The donor is dropped (`n_donors` 11 to 10). |
| P3 | `sp.ologit("y ~ x")` with string levels low / mid / high | Levels are ordered alphabetically with no warning. The slope is -0.43. With an ordered `Categorical` it is +1.11. |
| P4 | `inf` in a regressor or outcome | `regress` raises. `feols` drops the row and warns. `rdrobust` drops it silently (n 1000 to 999). `callaway_santanna` and `synth` keep going with numpy `RuntimeWarning`s, and the estimate changes. `dml` raises a raw sklearn `ValueError`. |
| P5 | `sp.bayes_regress` with 5 NaN rows | Rows dropped with no warning (`n_obs` 195). Listwise-deletion warnings exist in `regress`, `feols`, `rdrobust`, `dml` and `callaway_santanna`. |
| P6 | `sp.feols` on zero rows, all-NaN regressor, p > n, int64 near 2^62 | The pyfixest fallback leaks its own errors. They are "All variables are collinear", `ZeroDivisionError` and `LinAlgError: Matrix is singular`. `sp.regress` gives a taxonomy error with a clear message in each case. |
| P7 | `sp.regress` with a `Decimal` object column | Read as a 200-level factor. The error is "more parameters than observations". |
| P8 | `sp.regress` with int64 values near 2^62 | Float conversion collapses them. The column is then reported as "constant", with no mention of precision loss. |
| P9 | `sp.causal_forest` with a constant string effect modifier | Accepted silently. The ATE moves from 0.873 to 0.913. |
| P10 | `sp.arima` on a monthly `Series` with two dates removed | Fitted with no warning (implicit missing values). A shuffled date index is refused. `sp.var` accepts a shuffled date index. |
| P11 | `sp.arima` order (1,0,0) on a random walk | AR coefficient 0.987, no near-unit-root warning. |
| P12 | `result.to_excel("tab_x")`, `to_excel("tab.docx")` | No suffix is added. An xlsx file is written under a `.docx` name. |
| P13 | `sp.logit(maxiter=1)` | Warns. `model_info` has `n_iter` but no `converged` flag. `sp.glm` has `converged`. |
| P14 | Default seeds | `sp.dml` is reproducible by default and records `seed` in `result_card()["provenance"]`. `sp.callaway_santanna(bstrap=True)` and `sp.bayes_regress` are not reproducible without a seed. The `bayes_regress` result has no provenance record. |

---

## General standards

| ID | Standard (paraphrase) | Status | Evidence | Gap note |
| --- | --- | --- | --- | --- |
| G1.0 | List at least one primary reference from the literature | met | `paper.bib` master; `sp.citation()`; JOSS DOI 10.21105/joss.10604; `.cite()` on results; registry `reference` field | 403 certified / validated functions have an empty `reference` (e.g. `sp.var`) |
| G1.1 | Say whether it is a novel algorithm, a first implementation in the language, or an improvement on others | partial | README "Compared With Other Python Packages"; `docs/guides/migration-from-r.md`; parity notes name the R / Stata reference | No per-function statement of novelty or prior art |
| G1.2 | Life-cycle statement | partial | Per-function `stability` and `api_contract` (`docs/guides/stability.md`); classifier "Development Status :: 4 - Beta" | No package-level statement in README or CONTRIBUTING |
| G1.3 | Define statistical terms unambiguously | partial | Docstrings define estimands; `docs/guides/standard_errors.md`; `result.inference_rule()` | No glossary. Terms like CI, p-value and HC types are defined per function, not centrally |
| G1.4 | Document all functions in the standard tool | partial | `scripts/help_coverage.py`: docstrings 1680/1680, Examples 1680/1680, NumPy `Parameters` 1139/1680, `Returns` 1050/1680; `sp.function_schema` describes every parameter for 1640/1680 | A third of docstrings lack the NumPy `Parameters` section |
| G1.4a | Document internal functions too | partial | AST count over `src/statspai`: 4162 of 6946 private functions (60%) have docstrings | |
| G1.5 | Ship the code behind performance claims in publications | met | `tests/perf/*.py` and `*.R`, `tests/perf/compare_perf.py`; `tests/r_parity/compare.py`; `tests/coverage_monte_carlo/run_b1000.py` | `Paper-JSS/` is git-ignored here and lives in its own repo |
| G1.6 | Code to compare performance with other implementations | met | `tests/perf/01_hdfe_perf.R` to `04_dml_perf.R` against fixest, did, Synth, DoubleML; `docs/guides/performance.md` | |
| G2.0 | Assert input lengths (single vs multi-valued) | partial | `rdrobust` refuses `y=[...]`, `c=[0, 0.1]` (probe) | `sp.regress(5, df)` raises a raw `TypeError` |
| G2.0a | Document length expectations | met | `sp.function_schema` types and descriptions for 1640/1680 functions | |
| G2.1 | Assert input types | partial | `rdrobust` refuses non-integer `p`, negative `h`, `alpha` outside (0,1) (probe); `regress` refuses an ndarray `data` | Type checks are per estimator. Unhandled cases surface as raw `TypeError` / `ValueError` (P4, P6) |
| G2.1a | Document types of vector inputs | met | Registry `ParamSpec` types; `scripts/dump_schemas.py`; pre-push `registry-params` | |
| G2.2 | Refuse multivariate input to univariate parameters | met | `rdrobust(c=[0, 0.1])` raises "`c` must be a finite number" (probe) | |
| G2.3 | Univariate character input rules (below) | partial | see G2.3a, G2.3b | |
| G2.3a | Restrict to expected values (`match.arg` equivalent) | met | `Literal` / `src/statspai/_enum_seeds.py`; unknown `kernel`, `method`, `estimator`, `vce` are refused with the allowed list (probe) | |
| G2.3b | Case-insensitive, or documented as case-sensitive | partial | `rdrobust(kernel="Triangular")`, `synth(method="Classic")`, `dml(model="PLR")`, `bayes_regress(model="Normal")` accepted | `callaway_santanna(estimator="DR")` is refused, and the docstring does not say values are case-sensitive |
| G2.4 | Convert between data types appropriately | partial | see G2.4a to e | |
| G2.4a | Conversion to integer | partial | Nullable `Int64` accepted; int8 to int32 widened and overflow in `I(x**3)` handled (`core/_formula_dtypes.py`, probe) | String-coded time in `callaway_santanna` gives a raw `TypeError` |
| G2.4b | Conversion to continuous | partial | `boolean`, `Int64`, `float32` accepted (probe) | `Decimal` object column read as a factor (P7) |
| G2.4c | Conversion to character via `as.character`, not `paste` | n/a | R idiom about `paste`. No Python analogue risk | |
| G2.4d | Conversion to factor | met | Strings and `Categorical` become indicator terms; `@_expands_categorical` for covariate lists (`docs/dev/modules/formulas_covariates.md`) | |
| G2.4e | Conversion from factor | gap | | String levels in `ologit` are ordered alphabetically, silently (P3) |
| G2.5 | Say whether factor inputs must be ordered, and enforce it | gap | `sp.ologit` docstring says "Ordered categorical dependent variable" only | Unordered strings accepted; wrong sign with no warning (P3) |
| G2.6 | Pre-process 1-D input regardless of class | met | `timeseries/_forecast_common.py::read_series` takes list, ndarray, `Series`, one-column frame, or column name | |
| G2.7 | Accept many tabular forms | met | `src/statspai/_frame_input.py` (pandas, polars DataFrame / LazyFrame, Arrow, PyCapsule, interchange); `tests/test_frame_input.py`; polars probe matched pandas | |
| G2.8 | Convert once so internals see one type | met | `_frame_input.install` wraps every `sp.*` entry and hands pandas to the estimator | Column-level dtype handling is still per estimator |
| G2.9 | Warn when conversion loses or adds information | partial | Listwise-deletion `AssumptionWarning` with counts by column (`regress`, `feols`, `rdrobust`, `dml`) | Silent in P3, P7, P8, P9 |
| G2.10 | Column extraction behaves the same for every input class | met | All inputs become pandas before estimation (G2.8) | |
| G2.11 | Handle non-standard column classes, and test it | partial | Nullable `Int64` with `pd.NA` works in `regress` and `feols` (probe) | `Decimal` object column mis-handled (P7). No test found |
| G2.12 | Handle list columns (remove, convert, or informative error), and test it | partial | `regress` raises "y and X must be numeric arrays" | `dml` raises raw "setting an array element with a sequence". No test found |
| G2.13 | Check for missing data before analysis | partial | `statspai/_input_validation.py::clean_frame` (used in 262 source files); listwise-deletion warning in 13 | Silent in `synth` (P1, P2) and `bayes_regress` (P5) |
| G2.14 | Offer options for handling missing data (below) | partial | see G2.14a to c | |
| G2.14a | Option to error on missing data | gap | `sp.arima` always errors (good default) | No sampled cross-section estimator has a `missing="raise"` style option. Signature scan finds `missing=` only on 7 helpers |
| G2.14b | Option to ignore with a warning | partial | Default listwise deletion with warning in most sampled estimators | Silent in P2, P5 |
| G2.14c | Option to impute | partial | `sp.mice`, `sp.mi_estimate`, `sp.mi_test` (`src/statspai/imputation/`) | Separate workflow. No in-call option |
| G2.15 | Never assume non-missingness | partial | `regress`, `feols`, `rdrobust`, `dml`, `callaway_santanna` drop and report | `synth` passes NaN to the solver and reports `converged=True` (P1) |
| G2.16 | Options to handle NaN / Inf / -Inf | gap | | No option, and behaviour differs across estimators (P4) |
| G3.0 | No floating-point equality comparisons | partial | Test rule "do not compare degeneracy to 0" (`docs/dev/handbook/testing.md`); tolerance-based collinearity (`collin_tol`) | 138 exact `== 0.0` / `== 1.0` / `!= 0.0` comparisons in `src/statspai` (most are zero guards) |
| G3.1 | Let users choose the covariance algorithm | met | `vce` / `vcov` / `robust`: HC0 to HC4, cluster, CR2 / CR3, HAC, EWC, Conley, wild bootstrap (`sp.regress` error lists them) | |
| G3.1a | Document that choice | met | `docs/guides/standard_errors.md`; `scripts/se_menu_matrix.py` | |
| G4.0 | Add file suffixes when writing files | gap | | P12 |
| G5.0 | Test with standard data sets of known properties | met | `tests/numerical_accuracy/test_nist_strd_ols.py` and `test_nist_strd_anova.py` (all 22 NIST StRD sets) | |
| G5.1 | Make test data available to users | met | `sp.list_datasets()` (20 sets); `tests/r_parity/data/*.csv` in the repo and sdist | |
| G5.2 | Test error and warning behaviour | partial | `tests/test_actionable_errors.py`, `test_input_validation.py`, `test_core_degenerate_inputs.py`; taxonomy audit counts 5676 `MethodIncompatibility` and 1161 `DataInsufficient` raises | Not every message is tested (G5.2b) |
| G5.2a | Every message is unique | gap | | Not enforced. For example, "alpha must be in (0, 1)" appears 86 times in `src/statspai` in several spellings |
| G5.2b | Tests trigger every message | partial | Failure modes are generated from `recovery_hint` raises (`scripts/gen_source_failure_modes.py`) | No coverage map from message to test |
| G5.3 | Test that outputs have no NaN / Inf where none are expected | partial | `tests/test_mcp_nan_inf.py`, `test_silent_degradation_fixes.py` | P1 shows a NaN loss inside a result marked converged |
| G5.4 | Correctness tests against fixed data | met | `sp.parity_summary()`: 508 functions cross-language, 985 of 1012 estimators verified; `docs/parity.md`; `tests/r_parity/`, `tests/stata_parity/` | |
| G5.4a | New methods tested on trivial cases or multiple implementations | met | Rust vs NumPy HDFE (`docs/dev/handbook/rust.md`); analytic DGPs in `tests/reference_parity/` | |
| G5.4b | Test against previous implementations | met | Track A same-byte parity at 1e-6 vs R / Stata (`tests/r_parity/compare.py::TOLERANCES`) | |
| G5.4c | Use published numbers where no code exists | met | `tests/reference_parity/REFERENCES.md`; `sp.list_replications()` | |
| G5.5 | Run correctness tests with a fixed seed | met | Goldens regenerate to 1e-9 (`docs/dev/handbook/parity.md`); `tests/test_seed_contract.py` | |
| G5.6 | Parameter recovery tests | met | 672 `test_*.py` files in `tests/reference_parity/` (known-truth DGPs and frozen references) | |
| G5.6a | Recovery within a tolerance | met | Tolerances stated next to each assertion (handbook rule) | |
| G5.6b | Recovery over multiple seeds | partial | `tests/coverage_monte_carlo/` B=1000 for 12 DGPs (`FINDINGS.md`) | Most recovery tests use a single seed |
| G5.7 | Performance tests as data change (accuracy with n, time) | partial | `tests/perf/` log-log timing; `tests/reliability/` (few clusters, extreme weights, mass points) | No systematic "error shrinks with n" tests |
| G5.8 | Edge-condition tests (below) | partial | `tests/test_core_degenerate_inputs.py`, `test_cov95_synth_edges.py`, `test_iv_cov_edges.py` | Probes found untested edge paths (P1, P4, P6, P7) |
| G5.8a | Zero-length data | partial | `regress`, `rdrobust`, `callaway_santanna`, `dml`, `arima`, `bayes_regress` give clear errors | `feols` gives "All variables are collinear" (P6) |
| G5.8b | Unsupported types | partial | `rdrobust` and `feols` refuse string outcomes clearly | `regress` string outcome gives "y has 40000 rows but X has 200"; `arima` strings give a raw `ValueError` |
| G5.8c | All-NA or all-identical columns | partial | Constant regressor omitted with a note (`regress`, `feols`); constant series refused (`arima`) | `feols` all-NaN gives a misleading message; `rdrobust` constant outcome returns SE 0 with only a `RuntimeWarning` |
| G5.8d | Data outside algorithm scope (p > n) | partial | `regress` raises `DataInsufficient` with nobs and parameter count | `feols` raises `ZeroDivisionError` (P6) |
| G5.9 | Noise susceptibility tests (below) | partial | see G5.9a, b | |
| G5.9a | Machine-eps noise does not change results | partial | Probe on `regress` showed identical coefficients | No systematic test found |
| G5.9b | Different seeds do not change results meaningfully | partial | Monte Carlo coverage; `tests/test_estimator_determinism.py` (8 tests) | Not run per stochastic estimator |
| G5.10 | Extended tests behind a switch | met | `slow` marker excluded by default (`pyproject.toml` `addopts`); weekly `.github/workflows/slow-tests.yml`; `STATSPAI_FETCH_DATASETS` | Python marker in place of an env var |
| G5.11 | Large assets downloaded in the test workflow | met | `datasets/_fetch.py` with pinned SHA-256 (`docs/dev/modules/datasets.md`) | |
| G5.11a | Skip, not fail, when a download fails | met | Tests skip with a reason when data is not cached and the env var is unset | |
| G5.12 | Document conditions for extended tests | partial | Marker help text; `slow-tests.yml` comments; `tests/numerical_accuracy/README.md`, `tests/reliability/README.md`; `docs/dev/handbook/testing.md` | `CONTRIBUTING.md` does not mention slow, parity or download tests, or their runtime |

## Regression and supervised learning

| ID | Standard (paraphrase) | Status | Evidence | Gap note |
| --- | --- | --- | --- | --- |
| RE1.0 | Formula interface | met | `sp.regress`, `sp.feols`, `sp.logit`, `sp.glm`, `sp.ologit`, `sp.bayes_regress` take formulas; R and Stata idioms translated (`core/utils.py::r_formula_idioms`) | |
| RE1.1 | Document how formulas become matrices | partial | `docs/dev/modules/formulas_covariates.md` (developer notes); `docs/guides/migration-from-statsmodels.md` | No user page on coding, reference levels and term expansion |
| RE1.2 | Document accepted and rejected predictor types | partial | Schema types; categorical rules in `formulas_covariates.md` | Rejected types are not listed (P7) |
| RE1.3 | Keep row and column names on outputs | partial | Coefficient names kept; `data_info["sample_index"]` on `regress` | `residuals` and `fitted_values` are bare ndarrays with no index, so they are hard to align after listwise deletion |
| RE1.3a | Document what is not carried over | gap | | Not documented |
| RE1.4 | Document input assumptions, and test violations | partial | Registry `assumptions` on 983 of 1680 functions; `failure_modes` on 976 | |
| RE2.0 | Document transformations, and allow opting out | partial | Boolean outcome folding, `I()` power rewrite, categorical expansion documented in `formulas_covariates.md` | Automatic expansion has no documented opt-out |
| RE2.1 | Explicit missing-value parameters, NA distinct from Inf | gap | | No parameter (G2.14a, G2.16) |
| RE2.2 | Different missing handling for predictors and response | partial | `predict(newdata=...)` predicts rows whose outcome is missing | No separate policy at fit time |
| RE2.3 | Centring / offset options | partial | `glm(offset=, exposure=)`; `sp.demean`; weakly informative prior centres internally | No centring parameter on `regress` / `feols` |
| RE2.4 | Detect perfect collinearity (below) | met | see RE2.4a | |
| RE2.4a | Among predictors | met | `regress` omits with a note naming the combination, `collinear="raise"` to fail; `feols` `collin_tol`; `bayes_regress` refuses with the rank (probe) | |
| RE2.4b | Between predictors and response | gap | | `regress("yy ~ x1 + x2")` with `yy = 3*x1 + 1` fits with no warning |
| RE3.0 | Warn on non-convergence | met | `logit(maxiter=1)` and `glm(maxiter=1)` warn (P13) | |
| RE3.1 | Warnings can be suppressed, but the result still shows non-convergence | partial | Python warning filters suppress; `glm` stores `converged` | `logit` stores only `n_iter` (P13). `synth` reports `converged=True` with a NaN loss (P1) |
| RE3.2 | Sensible documented default thresholds | met | `logit` / `glm` docstrings: `tol=1e-8`, `maxiter=100` with meaning stated | |
| RE3.3 | Allow setting thresholds | met | `tol`, `maxiter` parameters | |
| RE4.0 | Return a model object | met | `EconometricResults`, `CausalResult`, domain result classes | |
| RE4.1 | May allow a model object without fitting | gap | | Not offered. Optional standard ("may") |
| RE4.2 | Coefficients accessor | met | `.params`, `.coef()` | |
| RE4.3 | Confidence intervals accessor | met | `.conf_int()` returns a DataFrame | |
| RE4.4 | Model specification (formula) accessor | partial | `model_info["formula"]` on `feols`, `logit`, `glm`; `.formula` on `bayes_regress` | `regress` does not store the formula |
| RE4.5 | Number of observations | met | `.nobs`, `.n_obs` | |
| RE4.6 | Variance-covariance matrix | met | `.vcov()`, `.cov_params()` | |
| RE4.7 | Convergence statistics | partial | `glm`: `converged`, `n_iter`; `logit`: `n_iter` | Not uniform |
| RE4.8 | Response variable and metadata | partial | `data_info["y"]`, `data_info["dependent_var"]` | Internal dict keys, not a documented accessor |
| RE4.9 | Fitted values | met | `.fitted_values` | |
| RE4.10 | Residuals, documented | partial | `.residuals` | No index; residual type (raw, Pearson, deviance) not stated for `regress` |
| RE4.11 | Goodness of fit and effect sizes | met | `.r2`, `.r2_adj`, `.glance()`, AIC / BIC / pseudo-R2 in `model_info` | |
| RE4.12 | Transform functions and inverses | n/a | Regression entries do not transform inputs beyond the formula terms the user writes. Time-series transforms are under TS4.5 | |
| RE4.13 | Predictor variables and metadata | partial | `data_info["X"]`, `data_info["var_names"]` | Internal keys |
| RE4.14 | Extrapolation / forecast errors | met | `predict(what=..., alpha=)`; `arima.forecast` returns lower / upper | |
| RE4.15 | Show that forecast errors grow with horizon | partial | R parity of ETS intervals (`tests/reference_parity/test_forecasting_r_parity.py`) | No explicit widening assertion |
| RE4.16 | Accept new groups in `predict()` | partial | Unseen factor levels are refused on purpose (`docs/dev/handbook/cross_cutting.md`, `reject_unseen_levels`) | Deliberate departure. No new-group prediction for mixed models found |
| RE4.17 | `print` shows inputs and coefficients | partial | `repr` is a one-line summary (`<EconometricResults: OLS, 3 parameters, 45 observations>`); `.summary()` is full | |
| RE4.18 | `summary` method | met | `.summary()`, `.tidy()`, `.glance()` | |
| RE5.0 | Document scaling of speed with data size | met | `docs/guides/performance.md`; `tests/perf/compare_perf.py` log-log figure | |
| RE6.0 | Default plot method | met | `.plot()` on `EconometricResults` and `CausalResult` | |
| RE6.1 | Direct users if plot is not dispatched on class | n/a | `.plot()` is a method on the result class | |
| RE6.2 | Default plot shows fitted values with intervals | gap | | `regress(...).plot()` draws a coefficient plot |
| RE6.3 | Forecast plots separate fitted and forecast | met | `arima.plot` / `ets.plot` legends: observed, fitted, forecast | |
| RE7.0 | Tests with noiseless exact relations among predictors | met | `tests/test_ols.py`, `test_regression_scan_oct2026_ols.py` assert "omitted because of collinearity" | |
| RE7.0a | Confirm noiseless predictors are rejected | met | Same tests; `collinear="raise"` | |
| RE7.1 | Tests with exact predictor-response relations | partial | Exact-fit SE rule in `docs/dev/handbook/testing.md`; NIST exact-fit designs | |
| RE7.1a | Exact-fit fitting is no slower than noisy fitting | gap | | No test |
| RE7.2 | Test that outputs keep row / case names | partial | `sample_index` exists | No test found |
| RE7.3 | Test accessor methods on results | met | `tests/test_postestimation_contract.py`, `test_did_result_accessors.py`, `scripts/result_protocol_audit.py` | |
| RE7.4 | Test that forecast errors grow with horizon | partial | as RE4.15 | |

## Bayesian and Monte Carlo

Scope is `statspai.mcmc` (NumPy samplers, `sp.bayes_regress` sampled in depth). The PyMC wrappers in `statspai.bayes` are noted where they differ.

| ID | Standard (paraphrase) | Status | Evidence | Gap note |
| --- | --- | --- | --- | --- |
| BS1.0 | Define "hyperparameter" where used | met | `mcmc/gp.py` names them (`length_scale[...]` per regressor) | |
| BS1.1 | Describe how to enter data, simple and multi-input | met | `docs/guides/bayesian_econometrics.md`; formula interface | |
| BS1.2 | Describe how to specify priors (below) | partial | see BS1.2a to c | |
| BS1.2a | In the README | gap | | README never mentions priors |
| BS1.2b | In a vignette | met | `docs/guides/bayesian_econometrics.md` (prior conventions, `prior_var`, `sigma2_prior`) | |
| BS1.2c | In function docs | met | `sp.bayes_regress` docstring (`prior_mean`, `prior_var`, `sigma2_prior`, `prior=`) | |
| BS1.3 | Describe computational parameters | met | `draws`, `burnin`, `thin`, `chains`, `seed`, `tune` documented | |
| BS1.3a | Document reusing previous output as a start | gap | | Not supported (BS2.8) |
| BS1.3b | Document choosing sampling algorithms | partial | `inference="mcmc"` or `"vb"`; sampler fixed per `model` and documented | No user choice of sampler within a model |
| BS1.4 | Document use with and without convergence checkers | partial | Checks always run; `diagnostics()` documented | No "without" mode documented |
| BS1.5 | Test differences between multiple checkers | partial | Each checker matches R `coda` to 1e-9 (`docs/dev/modules/bayes_mcmc.md`, `tests/reference_parity/test_bayes_mcmc_parity.py`) | Differences between checkers not tested |
| BS2.1 | Ensure inputs are dimensionally commensurate | met | Formula design; prior length checks (`mcmc/_core.py::normal_prior`) | |
| BS2.1a | Test those routines | met | `tests/test_bayes_regress.py::test_bad_arguments_are_refused`, `test_degenerate_data_is_refused` | |
| BS2.2 | Validate distributional parameters before sampling | met | `normal_prior` Cholesky check; `sigma2_prior` positivity (probe) | |
| BS2.3 | No excess prior values silently dropped | met | `prior_mean` of wrong length refused, naming coefficients (probe) | |
| BS2.4 | Prior lengths match the model | met | as BS2.3 | |
| BS2.5 | Check variance / shape parameters are valid | met | Positive-definite `prior_var`, positive `sigma2_prior` (probe) | |
| BS2.6 | Computational parameters in plausible ranges | met | `_core.check_mcmc_args` (`draws=0`, `chains=2.5` refused, probe) | |
| BS2.7 | Control starting values, per chain | gap | | No `inits` parameter in `bayes_regress` |
| BS2.8 | Start from a previous run | gap | | Not supported |
| BS2.9 | Different seed per chain by default | met | `_core.spawn_rngs` uses `SeedSequence.spawn` | |
| BS2.10 | Warn on identical seeds across chains | n/a | Users pass one seed; per-chain streams are spawned, so identical chain seeds cannot be supplied | |
| BS2.11 | Plural name for vector starting values | n/a | No starting-value parameter exists (see BS2.7) | |
| BS2.12 | A verbosity parameter, verbose by default | gap | `statspai.bayes` has `progressbar` (default False) | `statspai.mcmc` samplers have none |
| BS2.13 | Suppress messages but keep warnings, tested | partial | `mcmc` emits no progress messages; `ConvergenceWarning` is its own class | No control and no test |
| BS2.14 | Suppress warnings, tested | partial | Standard `warnings` filters on `ConvergenceWarning` / `StatsPAIWarning`; text also kept in `diagnostics_info["warnings"]` | No test of suppression |
| BS2.15 | Errors can be caught and processed | met | Taxonomy exceptions (`sp.MethodIncompatibility`, `sp.DataInsufficient`) with recovery hints; tested | |
| BS3.0 | Document missing-value assumptions | gap | | Rows silently dropped (P5) |
| BS3.1 | Diagnose perfect collinearity | met | `bayes_regress` refuses "collinear (rank 3 of 4)" (probe) | |
| BS3.2 | Separate fast path for collinear data | partial | Refused before any sampling | No alternative routine |
| BS4.0 | Document sampling algorithms with citations | met | Docstring names Albert and Chib (1993), Chib (1992), Kozumi and Kobayashi (2011), with bib keys | |
| BS4.1 | Compare with external samplers | partial | Seed screens against MCMCpack, bayesm, stochvol, dbarts (S level) in `bayesian_econometrics.md` | No stated advantage benchmark |
| BS4.2 | At least one way to validate posteriors | met | Exact-posterior grids, enumeration, prior importance sampling, joint-distribution tests (`tests/reference_parity/test_bayes_*_exact_posterior.py`) | |
| BS4.3 | Offer a convergence checker with a reference | met | `gelman_rubin`, `geweke_diag`, `heidel_diag`, `raftery_diag`, `mcmc_ess` (coda parity) | |
| BS4.4 | Option to stop on convergence | gap | | Fixed draw counts only |
| BS4.5 | Mechanism for models that do not converge | met | Stops after `burnin + draws * thin`; `ConvergenceWarning` when ESS < 100, split R-hat > 1.05, or acceptance outside 0.1 to 0.7 (`mcmc/regress.py`) | |
| BS4.6 | Converged-stop results equal fixed-length results | n/a | No convergence-based stopping (BS4.4) | |
| BS4.7 | Test effects of checker parameters | partial | `gelman_rubin(confidence=0.9)` checked against coda | No test of threshold direction |
| BS5.0 | Return seeds / starting values | gap | `sp.dml` records `seed` in provenance (P14) | `BayesRegressResult` has no seed and no provenance |
| BS5.1 | Return metadata on input types and dimensions | partial | `n_obs`, `formula`; data hash on `regress` / `dml` result cards | No data hash or dtype record on `mcmc` results |
| BS5.2 | Return the prior specification | met | `result.prior` dict, `result.formula` | |
| BS5.3 | Return convergence statistics | met | `diagnostics_info` (`min_ess`, `max_split_rhat`), `diagnostics()` | |
| BS5.4 | Say which checker was used | met | `diagnostics()` keyed by checker name | |
| BS5.5 | Non-convergence diagnostics accessible | met | `diagnostics_info["warnings"]` keeps the text | |
| BS6.0 | Default print method | met | `repr` prints the posterior table with sampler, chains, draws, burn-in | |
| BS6.1 | Default plot method | met | `.plot(kind="trace" / "density" / "acf")` | |
| BS6.2 | Plot traces with burn-in marked | partial | Trace plot per chain | Burn-in draws are not kept, so they cannot be shown |
| BS6.3 | Plot posterior distributions | met | `.plot(kind="density")` | |
| BS6.4 | `summary` method | met | `.summary()`, `.tidy()` | |
| BS6.5 | Traces and densities in one graphic | met | `kind="trace"` draws trace and histogram side by side | |
| BS7.0 | Recover parametric prior estimates | partial | Joint-distribution tests (parameters marginally follow the prior) for `stochvol`, probit systems, `bart` | Not for every sampler |
| BS7.1 | Recover the prior with no data | partial | as BS7.0 | |
| BS7.2 | Recover the expected posterior given prior and data | met | Grid-integrated posteriors written from `scipy.stats` (`tests/reference_parity/test_bayes_regress_exact_posterior.py` and siblings) | |
| BS7.3 | Test scaling of efficiency with data size | gap | | No test |
| BS7.4 | Test fitted values are on the input scale | not checked | | |
| BS7.4a | Test implications of input scale assumptions | partial | `test_default_prior_that_is_not_vague_is_flagged` (scale note on the default prior) | |

## Time series

Scope is `statspai.timeseries` (`arima`, `ets`, `var`, `corrgram`, `periodogram`), sampled.

| ID | Standard (paraphrase) | Status | Evidence | Gap note |
| --- | --- | --- | --- | --- |
| TS1.0 | Rely on explicit time-series classes; refuse generic input | partial | Uses `DatetimeIndex` / `PeriodIndex` when given (`read_series`) | Bare arrays are accepted. The Python ecosystem has no single `ts` class, so this is a design choice |
| TS1.1 | Document accepted input classes | met | `arima` docstring and `read_series` (array, `Series`, one-column frame, column with `data=`) | |
| TS1.2 | Validate input classes | partial | Multi-column frame and bad column name refused | String series gives a raw `ValueError` |
| TS1.3 | One pre-processing routine for all functions | partial | `timeseries/_forecast_common.py::read_series` | Imported by 11 of 64 modules; `var` is not covered (P10) |
| TS1.4 | Keep time attributes of the input | partial | `forecast()` returns future dates as the index | `fitted_values` and `residuals` are bare arrays |
| TS1.5 | Strict ordering of the time index | partial | `require_time_order` refuses unsorted, duplicate or NaT dates | Only where `read_series` is used |
| TS1.6 | Catch ordering violations in pre-processing | partial | `arima` refuses a shuffled index (P10) | `var` accepts it (P10) |
| TS1.7 | Accept `units`-package input | n/a | R `units` package. No equivalent is standard for pandas time series | |
| TS1.8 | State the calendar convention for days / months | gap | | Not documented; relies on pandas frequency rules |
| TS2.0 | Refuse or flag implicit missing values in regular series | gap | | Dropped dates are fitted silently (P10) |
| TS2.1 | Options for missing data (below) | partial | see TS2.1a to c | |
| TS2.1a | Error on missing data | met | Interior NaN refused with a recovery hint (`read_series`, probe) | |
| TS2.1b | Warn and treat as irregular, with equal results | gap | | Not offered |
| TS2.1c | Impute | gap | Recovery hint suggests `Series.interpolate()` | No option |
| TS2.2 | Consider stationarity of first and second moments | partial | Unit-root tests (`sp.unitroot` family, KPSS in `ndiffs`); GARCH family | Not considered inside `arima` with a fixed order |
| TS2.3 | Document stationarity assumptions | partial | `arima` docstring (`enforce_stationarity`, KPSS rule for `d`) | |
| TS2.4 | Check stationarity (below) | partial | see TS2.4a, b | |
| TS2.4a | Warn | partial | `_near_unit_root` guards the automatic search | No warning for a fixed order (P11) |
| TS2.4b | Enable or advise transformations | met | `auto=True` picks `d` by KPSS; `sp.ndiffs`, `sp.nsdiffs`, `boxcox=` | |
| TS2.5 | Auto-covariance rows follow the series order | met | `sp.corrgram` returns a lag-indexed table | |
| TS2.6 | Units on auto-covariance matrices | n/a | No units system in use (TS1.7) | |
| TS3.0 | Test a case where errors widen with horizon | partial | ETS intervals match R `forecast` (`test_forecasting_r_parity.py`) | No explicit widening assertion |
| TS3.1 | Test a case that violates TS3.0 | met | `tests/test_forecasting_toolkit.py::test_ets_multiplicative_error_intervals_widen_with_the_level` (widest interval on the seasonal peak) | |
| TS3.2 | Document what drives forecast errors | partial | `docs/guides/forecasting_fpp.md` (bootstrap intervals, `dof_adjust`, back-transform) | No section on drivers |
| TS3.3 | Trim forecasts to an error margin (below) | gap | | |
| TS3.3a | Document how to trim | gap | | |
| TS3.3b | Provide a trimming mechanism | gap | | |
| TS4.0 | Return same class as input, or a defined class (below) | met | see TS4.0b | |
| TS4.0a | Same class as input | n/a | Alternative to TS4.0b, which is met | |
| TS4.0b | A unique class-defined format | met | `ARIMAResult`, `ETS` result, `VARResult` | |
| TS4.1 | Carry input units into outputs | n/a | No units system (TS1.7) | |
| TS4.2 | Document return types | met | Return annotations and `Returns` sections (`arima` -> `ARIMAResult`, `forecast` -> `pd.DataFrame`) | |
| TS4.3 | Include time scales in non-series returns | partial | Forecast index carries dates | `fitted_values` / `residuals` do not |
| TS4.4 | Document effect of stationarity transforms on forecasts | met | `forecasting_fpp.md` (back-transformed log forecast is the median); `biasadj=` | |
| TS4.5 | Back-transform (below) | met | see TS4.5a | |
| TS4.5a | Routines to back-transform | met | `_forecast_common.back_transform_frame`, `boxcox_inverse`, `biasadj` | |
| TS4.5b | Demonstrate back-transform | n/a | Alternative to TS4.5a, which is met | |
| TS4.5c | Document limits | n/a | Alternative to TS4.5a, which is met | |
| TS4.6 | Forecast return content (below) | met | see TS4.6c | |
| TS4.6a | A distribution object | n/a | Alternative; TS4.6c is met | |
| TS4.6b | First and second moments | partial | Point forecast returned | No SE column, only interval bounds |
| TS4.6c | General error indication | met | `forecast()` returns `lower` / `upper` at `level=` | |
| TS4.7 | Separate forecast from observed values (below) | met | see TS4.7a | |
| TS4.7a | Return forecasts alone | met | `forecast()` returns only future rows | |
| TS4.7b | Distinct list items | n/a | Alternative to TS4.7a | |
| TS4.7c | One object with a flag column | n/a | Alternative to TS4.7a | |
| TS5.0 | Default plot methods | met | `.plot()` on `ARIMAResult`, ETS, `periodogram` results | |
| TS5.1 | Label one axis as time, with continuous units | partial | `ets.plot` uses dates on the x axis | `arima.plot` labels the axis `t` and plots positions 0..n even with a `DatetimeIndex` |
| TS5.2 | Time on the horizontal axis | met | Both plots | |
| TS5.3 | Print time units on the axis | partial | `ets.plot` shows dates (no axis title) | `arima.plot` shows integers |
| TS5.4 | Frequency axes use [0, 0.5] or [0, 2 pi], labelled | met | `spectral.py` plots cycles per observation, with a radians option, labelled | |
| TS5.5 | Option for broken vs continuous lines with missing data | n/a | Interior missing values are refused (TS2.1a), so plots never contain them | |
| TS5.6 | Show forecast limits by default | met | Shaded interval in `arima.plot` | |
| TS5.7 | Include observed values with forecasts | met | "observed" series in the legend | |
| TS5.8 | Distinguish observed from forecast clearly | met | Separate labelled lines | |

---

## Prioritised gap list

The order is by harm to a specialist user. The first items return a wrong number with no signal, which a referee will treat as disqualifying. Later items are missing conveniences. Sizes are S (under a day), M (a few days), L (a week or more, or touches many estimators).

| # | Gap (IDs) | Impact on a specialist | Fix | Size |
| --- | --- | --- | --- | --- |
| 1 | `sp.synth` passes missing values into the solver (G2.13, G2.15, RE3.1; P1, P2) | One NaN in the treated pre-period gives equal donor weights, `converged=True` and a placebo p-value. Nothing on screen says the fit failed | Check panel completeness in synth pre-processing. Refuse NaN in the treated unit. Warn and record in `diagnostics` when a donor or a row is dropped. Set `converged=False` when the loss is not finite. Add tests for both cases | S |
| 2 | `sp.ologit` orders string outcome levels alphabetically (G2.4e, G2.5; P3) | Slope has the wrong sign with no warning. Survey data often arrive as strings | Refuse unordered string or object outcomes, or warn and print the order used. Accept ordered `Categorical` and numeric codes. Apply the same rule to `oprobit` and `mlogit` base level | S |
| 3 | Non-finite values handled five different ways (G2.16, RE2.1; P4) | The same data give a refusal, a warning, a silent drop or a changed estimate, depending on the estimator | Extend `_input_validation.clean_frame` to count `inf` separately from NaN, and apply one policy (drop with an `AssumptionWarning` that names the column, or raise). Route `rdrobust`, `callaway_santanna`, `synth`, `dml` through it | M |
| 4 | No user control of missing-data handling (G2.14a, RE2.1, RE2.2) | Users cannot ask for a hard failure on missing data, which careful replication work needs | Add a shared `missing="drop" / "raise"` option through the frame wrapper or `clean_frame`, register it as a `ParamSpec`, and test it on the core estimators first | L |
| 5 | `sp.feols` fallback leaks pyfixest errors on degenerate input (G5.8a, G5.8c, G5.8d; P6) | Zero rows, p > n or an all-NaN column give "All variables are collinear" or `ZeroDivisionError`. `sp.regress` gets these right | Run the same validation as `regress` (row count, parameter count, all-NaN, finite values) before backend dispatch | S |
| 6 | `statspai.mcmc` results drop NaN rows silently and keep no seed or provenance (BS3.0, BS5.0, BS5.1, G2.14b; P5, P14) | A posterior cannot be tied to its sample or rerun exactly from the result alone | Emit the same listwise-deletion warning as `regress`. Attach the standard provenance record (seed, data hash, shape) that `sp.dml` already has. Document the missing-value rule in the docstring | S |
| 7 | Time-series inputs with date gaps, and `var` without order checks (TS2.0, TS1.3, TS1.6; P10) | A monthly series with a missing month is fitted as if consecutive. A shuffled frame gives a VAR on scrambled time | In `read_series`, infer the index frequency and refuse or warn on gaps. Route `var`, `garch` and the rest of `timeseries/` through `read_series` | M |
| 8 | Silent type coercions (G2.9, G2.11; P7, P8, P9) | A `Decimal` column becomes a 200-level factor. An int64 ID near 2^62 is called "constant". A string effect modifier is accepted by `causal_forest` | Detect object columns holding numbers and convert them or refuse. Warn when an int64 column is not exactly representable in float64. Refuse non-numeric effect modifiers in `causal_forest` | S |
| 9 | Fitted values and residuals lose the row index; formula and convergence not uniform on results (RE1.3, RE4.4, RE4.7, RE4.10, TS1.4, TS4.3; P13) | After listwise deletion users cannot line residuals up with their data. `regress` does not record its formula. `logit` has no `converged` flag | Return `pd.Series` indexed by `sample_index` (or dates). Store `formula` in `model_info` for `regress`. Add `converged` to every iterative estimator. Pin it with a result-protocol test | M |
| 10 | Exported files get no suffix check (G4.0; P12) | `to_excel("table")` writes a file Excel will not open by name. `to_excel("t.docx")` writes xlsx bytes under a Word name | Append the right suffix when none is given and refuse a mismatched one, in the shared exporters | S |
| 11 | MCMC has no starting values, no continuation and no verbosity control (BS2.7, BS2.8, BS1.3a, BS2.12) | Long or troublesome chains cannot be resumed or started from a known point | Add `inits=` (per chain), a `previous=` or `continue_from=` path, and `verbose=`. Store burn-in draws optionally so trace plots can mark them (BS6.2) | M |
| 12 | `arima` with a fixed order gives no near-unit-root warning (TS2.4a; P11) | An AR coefficient of 0.987 on a random walk passes without comment | Reuse `_near_unit_root` after a fixed-order fit and warn, with a pointer to `sp.ndiffs` | S |
| 13 | `arima.plot` ignores the date index (TS5.1, TS5.3) | Plots show 0..n instead of dates | Plot against the stored index; label the axis with the frequency | S |
| 14 | References missing on 403 certified / validated functions (G1.0, G1.1) | A reviewer cannot see the method source for a validated function such as `sp.var` | Fill the registry `reference` field from `paper.bib` keys, with the usual two-source check. Add a ratchet in `scripts/help_coverage.py` | M |
| 15 | No package life-cycle statement or glossary (G1.2, G1.3, G5.12) | Reviewers ask for both. Extended-test conditions are scattered | Add a short life-cycle paragraph to README and CONTRIBUTING, a glossary page in `docs/guides/`, and a testing section in CONTRIBUTING covering slow, parity and download tests | S |
| 16 | Edge-case and noise tests are not systematic (G5.8, G5.9a, G5.6b, RE2.4b, RE7.1a) | The probes in this audit found defects that the suite did not | Add one parametrised suite that runs the P1 to P11 inputs (zero rows, all-NaN, constant, collinear, `inf`, strings, eps noise, exact fit) over the core estimators, asserting a taxonomy error or a documented warning | M |
| 17 | `regress(...).plot()` is a coefficient plot (RE6.2) | Minor. The standard expects fitted values with intervals | Add `type="fitted"` and keep `coefplot` as the default, or document the choice | S |
| 18 | Error messages are not unique and not each tested (G5.2a, G5.2b) | Low for users. Mainly a review checkbox | Generate a message inventory from the taxonomy audit and flag duplicates in shared helpers. Accept repeated generic messages from one helper | L |
| 19 | No Bayesian scaling test, no forecast trimming (BS7.3, TS3.3) | Low | Add a timing test across n for `bayes_regress`. Document trimming of forecasts by interval width in the forecasting guide | S |

Items 1, 2, 5 and 8 change what the package returns on some inputs. Under CLAUDE.md §12 any change to returned numbers goes in CHANGELOG and MIGRATION with the correctness-fix flag. Items 1 and 2 should also be checked against the Track A modules that touch `synth` and `ologit` before any provenance trace is re-recorded.

## Limits of this audit

- Each "met" for an input-handling standard rests on the sampled estimators and the shared mechanism, not on all 1,680 functions.
- Only `statspai.mcmc` was probed for the Bayesian set. The PyMC wrappers in `statspai.bayes` were read, not run.
- BS7.4 was not checked.
- No code was changed. Probe scripts were kept outside the repository.
