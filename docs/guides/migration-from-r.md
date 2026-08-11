# Migrating from R to StatsPAI

A practical one-page map for researchers moving from R's causal inference
ecosystem to StatsPAI. Every R function listed here has an independent
Python re-implementation in StatsPAI, following the same statistical
methodology but exposed through a unified `sp.*` API.

```python
import statspai as sp
```

---

## Regression & Fixed Effects (`fixest`)

| R (`fixest`)                              | StatsPAI                                                          | Notes                                                        |
| ----------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| `feols(y ~ x1 + x2, data = df)`           | `sp.feols("y ~ x1 + x2", data=df)`                                | pyfixest-backed; same formula syntax                         |
| `feols(y ~ x \| firm + year, data = df)`  | `sp.feols("y ~ x \| firm + year", data=df)`                       | Two-way fixed effects                                        |
| `fepois(y ~ x \| firm, data = df)`        | `sp.fepois("y ~ x \| firm", data=df)`                             | HDFE Poisson / PPML for gravity models                       |
| `feglm(..., family = "binomial")`         | `sp.feglm(..., family="binomial")`                                | HDFE GLM                                                     |
| `etable(m1, m2, m3)`                      | `sp.regtable(m1, m2, m3)`                                         | Publication-quality regression tables (LaTeX / MD / HTML / Word / Excel) |
| `vcov(m, cluster = ~firm)`                | `vcov={"CRV1": "firm"}` in `sp.feols`                             | Cluster-robust SE                                            |
| `fixef(m)`                                | `m.fixef()` (on pyfixest result)                                  | Extract fixed-effect estimates                               |

For plain OLS without HDFE, `sp.regress("y ~ x", data=df)` gives a statsmodels-
compatible interface returning an `EconometricResults`. It exports the same way
as every other result — `.to_latex()` / `.to_html()` / `.to_markdown()` /
`.to_excel()` / `.to_word()` — see the
[exporting regression results](exporting-regression-tables.md) guide.

---

## Staggered DID (`did`, `fixest::sunab`, `didimputation`, `DIDmultiplegt`)

| R                                                       | StatsPAI                                                 |
| ------------------------------------------------------- | -------------------------------------------------------- |
| `did::att_gt(yname, tname, gname, idname, data)`        | `sp.callaway_santanna(data, y=, time=, first_treat=, group=)` |
| `did::aggte(obj, type = "dynamic")`                     | `sp.aggte(cs_result, type="dynamic")`                    |
| `did::aggte(obj, type = "group")`                       | `sp.aggte(cs_result, type="group")`                      |
| `did::ggdid(agg)`                                       | `sp.ggdid(agg_result)`                                   |
| `fixest::sunab(cohort, period)` inside `feols`          | `sp.sun_abraham(data, y=, time=, first_treat=, group=)`  |
| `didimputation::did_imputation(...)`                    | `sp.did_imputation(data, y=, time=, first_treat=, group=)` |
| `DIDmultiplegt::did_multiplegt(...)`                    | `sp.did_multiplegt(data, y=, group=, time=, treatment=)` |
| `DIDmultiplegt::did_multiplegt_dyn(...)`                | `sp.did_multiplegt_dyn(...)` (experimental MVP; not yet R-parity) |
| `bacondecomp::bacon(...)`                               | `sp.bacon_decomposition(data, y=, time=, treat=, id=)`   |
| —                                                       | `sp.etwfe(...)` — Wooldridge (2021) explicit API         |
| `HonestDiD::createSensitivityResults(...)`              | `sp.honest_did(cs_result, m_grid=...)` / `sp.breakdown_m(...)` |
| `HonestDiD::createSensitivityResults_relativeMagnitudes` | `sp.sensitivity_rr(cs_result, Mbar=...)`                 |
| `pretrends::pretrends(...)`                             | `sp.pretrends_power(...)` — same pre-test by default; `test="joint"` for the joint Wald version, which `sp.pretrends_test(...)` also reports |
| `pretrends::slope_for_power(...)`                       | `sp.pretrends_slope_for_power(result, target_power=0.5)` |
| —                                                       | `sp.stacked_did(...)`, `sp.ddd(...)`, `sp.continuous_did(...)`, `sp.cic(...)` |

One-call integrated report:

```python
report = sp.cs_report(data, y="y", t="t", g="g", i="id",
                      save_to="output/did_report")
# Produces .txt + .md + .tex + .xlsx + .png in one call
```

---

## Regression Discontinuity (`rdrobust`, `rddensity`, `rdmulti`, `rdhonest`)

| R                                             | StatsPAI                                               |
| --------------------------------------------- | ------------------------------------------------------ |
| `rdrobust::rdrobust(y, x, c = 0)`             | `sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct")` for canonical R parity |
| `rdrobust::rdplot(y, x)`                      | `sp.rdplot(y, x)`                                      |
| `rdrobust::rdbwselect(y, x)`                  | `bwselect="cct"` for R parity; default `mserd` is dependency-light |
| `rddensity::rddensity(x, c = 0)`              | `sp.rddensity(df, x="x", c=0)` for native Python parity with the default `rddensity` selector/test path; use `backend="r"` only when direct R execution is required |
| `rdrobust::rdplotdensity(x, c = 0)`           | `sp.rdplotdensity(df, x="x", c=0)`                     |
| `rdmulti::rdmc(...)`                          | `sp.rdmc(...)`                                         |
| `rdmulti::rdms(...)`                          | `sp.rdms(...)`                                         |
| `rdhonest::RDHonest(...)`                     | `sp.rd_honest(...)` (Armstrong-Kolesár honest CI)      |
| —                                             | `sp.rkd(...)` — Regression kink designs                |

---

## Synthetic Control (`Synth`, `gsynth`, `augsynth`, `synthdid`)

| R                                            | StatsPAI                                     |
| -------------------------------------------- | -------------------------------------------- |
| `Synth::synth(...)`                          | `sp.synth(...)`                              |
| `gsynth::gsynth(Y ~ D \| X, data)`           | `sp.gsynth(data, y=, treat=, unit=, time=)`  |
| `augsynth::augsynth(...)`                    | `sp.augsynth(...)`                           |
| `synthdid::synthdid_estimate(...)`           | `sp.sdid(...)`                               |
| —                                            | `sp.staggered_synth(...)` — Staggered SCM    |
| —                                            | `sp.robust_synth(...)` — Robust SCM          |

---

## Matching & Reweighting (`MatchIt`, `ebal`, `cobalt`)

| R                                        | StatsPAI                                            |
| ---------------------------------------- | --------------------------------------------------- |
| `MatchIt::matchit(..., method = "nearest")` | `sp.match(..., method="psm")`                    |
| `MatchIt::matchit(..., method = "cem")`  | `sp.match(..., method="cem")`                       |
| `MatchIt::matchit(..., method = "mahal")` | `sp.match(..., method="mahalanobis")`              |
| `ebal::ebalance(...)`                    | `sp.ebalance(...)`                                  |
| `cobalt::bal.tab(...)`                   | Built into `sp.match()` output: `result.balance`    |

---

## IV (`AER`, `ivmodel`, `ivreg`)

| R                                              | StatsPAI                                              |
| ---------------------------------------------- | ----------------------------------------------------- |
| `AER::ivreg(y ~ x \| z, data = df)`            | `sp.ivreg("y ~ x", instruments=["z"], data=df)`       |
| `ivmodel::LIML(...)`                           | `sp.liml(...)`                                        |
| `ivmodel::JIVE(...)`                           | `sp.jive(...)`                                        |
| `hdm::rlassoIV(...)`                           | `sp.lasso_iv(...)`                                    |
| `ivmodel::AR.test(...)` / Stata `weakiv`       | `sp.anderson_rubin_test(...)`                         |
| Stata `weakivtest` (Olea-Pflueger F)           | `sp.effective_f_test(..., vcov='HC1')`                |
| Lee et al. (2022) tF                            | `sp.tF_critical_value(F)` + 2SLS t-ratio              |
| `fixest::feols(y ~ 1 \| fe \| x ~ z)`          | `sp.feols("y ~ 1 \| fe \| x ~ z", data=df)`           |
| `MendelianRandomization::mr_ivw(...)`          | `sp.mr_ivw(...)`                                      |
| `MendelianRandomization::mr_egger(...)`        | `sp.mr_egger(...)`                                    |

---

## Machine Learning Causal Inference

| R                                            | StatsPAI                                            |
| -------------------------------------------- | --------------------------------------------------- |
| `DoubleML::DoubleMLPLR$new(...)`             | `sp.dml(..., model="plr")`                          |
| `DoubleML::DoubleMLIRM$new(...)`             | `sp.dml(..., model="irm")`                          |
| `grf::causal_forest(X, Y, W)`                | `sp.causal_forest(X, Y, W)`                         |
| `grf::causal_forest(...)$predict(...)`       | `forest.predict(X_new)`                              |
| `grf::instrumental_forest(...)`              | `sp.causal_forest(..., instrumental=True)`          |
| `SuperLearner::SuperLearner(...)`            | `sp.tmle(...)` with custom learners                 |
| `policytree::policy_tree(...)`               | `sp.policy_tree(X, reward)`                         |
| `causalTree::honest.causalTree(...)`         | `sp.causal_forest(..., honest=True)`                |

Meta-learner suite (no single R package covers all of these):

```python
# Unified one-call API:
sp.metalearner(data, y="y", treat="D", covariates=["x1", "x2"], learner="S")
sp.metalearner(data, ..., learner="T")
sp.metalearner(data, ..., learner="X")
sp.metalearner(data, ..., learner="R")
sp.metalearner(data, ..., learner="DR")  # default

# Or use the class API for finer control:
from statspai import SLearner, TLearner, XLearner, RLearner, DRLearner
```

---

## Bounds, Conformal, Discovery, Policy

| R                                      | StatsPAI                                            |
| -------------------------------------- | --------------------------------------------------- |
| `bounds::bounds(...)` (manual)         | `sp.manski_bounds(...)`, `sp.lee_bounds(...)`       |
| `EValue::evalues.OR(...)`              | `sp.evalue(...)`                                    |
| `pcalg::pc(suffStat, ...)`             | `sp.pc_algorithm(data)`                             |
| `pcalg::fci(...)`                      | `sp.causal_discovery(data, method="fci")`           |
| — (NOTEARS: Python only historically)  | `sp.notears(data)`                                  |
| `policytree::policy_tree(...)`         | `sp.policy_tree(...)`                               |
| —                                      | `sp.conformal_cate(...)` — conformal CATE           |

---

## Post-estimation, Diagnostics, Robustness

| R                                                | StatsPAI                                           |
| ------------------------------------------------ | -------------------------------------------------- |
| `modelsummary::modelsummary(list(m1, m2))`       | `sp.outreg2([m1, m2])` or `sp.modelsummary([m1, m2])` |
| `sandwich::vcovCL(m, cluster = ~id)`             | `vcov={"CRV1": "id"}` in `sp.feols` / `sp.regress` |
| `lmtest::coeftest(m, vcov = vcovHC)`             | `result.robust()` / `result.summary(vcov="HC3")`   |
| `car::linearHypothesis(m, "x1 = x2")`            | `result.test("x1 = x2")`                           |
| `marginaleffects::avg_slopes(m)`                 | `result.marginal_effects()`                        |
| `multiwayvcov::cluster.vcov(m, ~c1 + c2)`        | `vcov={"CRV1": ["c1", "c2"]}`                      |
| `specr::specr(...)` / `spec_curve`               | `sp.spec_curve(...)` — native implementation       |

---

## What StatsPAI Adds That R Cannot

These are not migration items — they are **net-new capabilities** that the
R ecosystem does not currently offer in unified form:

- **`sp.list_functions()` / `sp.describe_function()` / `sp.function_schema()`** — agent-native introspection for LLM workflows
- **`sp.interactive(fig)`** — Jupyter WYSIWYG plot editor with reproducible code export
- **`sp.cs_report(..., save_to=...)`** — one-call bundle: txt + md + tex + xlsx + png
- **Unified `CausalResult`** across all 390+ functions with `.summary()`, `.plot()`, `.to_latex()`, `.cite()`
- **sklearn Pipeline / JAX / PyTorch integration** — neural causal models natively

---

## Quick Install

```bash
pip install statspai
# Optional HDFE backend (for sp.feols / sp.fepois at scale):
pip install pyfixest
```

---

If you find a missing R function that should be mapped here, please open
an issue at [github.com/brycewang-stanford/statspai/issues](https://github.com/brycewang-stanford/statspai/issues).
