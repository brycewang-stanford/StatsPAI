# `sp.dml` and the DoubleML reference implementation

`sp.dml` is StatsPAI's implementation of the Double/Debiased Machine
Learning (DML) framework of Chernozhukov et al. (2018). The canonical
reference implementations are the [DoubleML][doubleml] R package
(Bach, Kurz, Chernozhukov, Spindler & Klaassen, JSS 108(3), 2024)
and the [doubleml-for-py][doubleml-py] Python package
(Bach, Chernozhukov, Kurz & Spindler, JMLR 23(53), 2022).

This guide covers (1) how the `sp.dml` API maps onto DoubleML, (2)
where the numbers agree, and (3) what the differences are and why
they exist.

[doubleml]: https://github.com/DoubleML/doubleml-for-r
[doubleml-py]: https://github.com/DoubleML/doubleml-for-py

## API mapping

`sp.dml` is a single dispatcher; the `model=` argument selects the
estimator and matches one of DoubleML's classes one-to-one.

| Model | `sp.dml(...)` | `doubleml.DoubleML*(...)` | DoubleML R |
| --- | --- | --- | --- |
| Partially linear regression | `sp.dml(model='plr', ml_g=..., ml_m=...)` | `DoubleMLPLR(ml_l=..., ml_m=...)` | `DoubleMLPLR$new(...)` |
| Interactive regression (AIPW ATE) | `sp.dml(model='irm', ml_g=..., ml_m=...)` | `DoubleMLIRM(ml_g=..., ml_m=...)` | `DoubleMLIRM$new(...)` |
| Partially linear IV | `sp.dml(model='pliv', instrument=..., ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLPLIV(ml_l=..., ml_m=..., ml_r=...)` | `DoubleMLPLIV$new(...)` |
| Interactive IV (LATE) | `sp.dml(model='iivm', instrument=..., ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLIIVM(ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLIIVM$new(...)` |

Anything that takes a scikit-learn estimator on the DoubleML side also
works in `sp.dml`. StatsPAI additionally accepts string shortcuts
(`'rf'`, `'gbm'`, `'lasso'`, `'ridge'`, `'linear'`, `'logistic'`,
`'xgb'`, `'lgbm'`) for the common nuisance learners.

## Same-DGP, same-seed numerical agreement

The fixture in `tests/reference_parity/_fixtures/dml_data.csv` is a
seed-42 DGP with `n=1000`, `p=10`, true treatment effect `θ=0.5`. The
external parity test
[`tests/external_parity/test_dml_python_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/external_parity/test_dml_python_parity.py)
runs `sp.dml` and `doubleml-for-py` on this fixture with identical
scikit-learn learners (`LassoCV(cv=5)` for regression,
`LogisticRegressionCV(cv=5)` for binary propensity) under a fixed
seed.

The non-instrumented models (PLR, IRM) use `dml_data.csv`; the
instrumented models (PLIV, IIVM) use the companion `dml_iv_data.csv`
(n=2000, with a continuous instrument `z_c` and a binary instrument
`z_b`; see `_generate_dml_iv_data.py`). All four DoubleML model classes
are pinned against `doubleml-for-py`.

| Model | `sp.dml` (StatsPAI 1.17.0) | `doubleml-for-py` 0.11.3 | `DoubleML` R 1.0.2 (cv.glmnet) |
| --- | --- | --- | --- |
| **PLR** (continuous d) | **+0.5590 ± 0.0331** | **+0.5590 ± 0.0331** | +0.5368 ± 0.0335 |
| **IRM** (binary d, AIPW) | -0.0191 ± 0.0766 | -0.0267 ± 0.0742 | +0.0066 ± 0.0744 |
| **PLIV** (continuous d, instrument `z_c`) | **+0.5117 ± 0.0195** | **+0.5117 ± 0.0195** | — (not pinned on R side) |
| **IIVM** (binary d, instrument `z_b`, LATE) | +0.5495 ± 0.0924 | +0.5618 ± 0.0919 | — (not pinned on R side) |

- **PLR**: `sp.dml` and `doubleml-for-py` agree to **machine precision**
  on both the point estimate and the standard error — |Δ| = 1.1 × 10⁻¹⁶
  on the coefficient and 1.4 × 10⁻¹⁷ on the standard error, i.e. one
  float64 unit in the last place. This is exact numerical equivalence
  under shared scikit-learn folds — both implementations evaluate the
  same Neyman-orthogonal score on the same CV-fold partition under a
  fixed seed. The slight deviation from the R reference (~4.1%) reflects
  glmnet's penalty path differing fractionally from scikit-learn's
  `LassoCV`; the R reference is pinned by
  [`tests/reference_parity/test_dml_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/reference_parity/test_dml_parity.py)
  to within 7% relative.

- **IRM**: All three implementations land statistically at zero (the
  truth for this DGP). `sp.dml` and `doubleml-for-py` differ by
  0.0076 absolute — about one-tenth of a standard error — owing to
  internal differences in how the two AIPW scores are constructed. This
  residual is verified *not* to come from propensity trimming (matching
  the clip thresholds leaves it unchanged) nor from IPW normalization
  (toggling `doubleml-for-py`'s `normalize_ipw` leaves it unchanged). The
  external parity test tolerates 0.05 absolute deviation, which is
  roughly two-thirds of one SE on this fixture.

- **PLIV**: Like PLR, the partially linear IV estimator residualises
  `y`, `d`, and the instrument `z_c` on `X` and evaluates the same
  partialling-out score on a shared `KFold` partition. `sp.dml` and
  `doubleml-for-py` agree to **machine precision** on both the
  coefficient (|Δ| = 0) and the standard error (|Δ| ~ 3 × 10⁻¹⁷).

- **IIVM**: The interactive-IV LATE estimator behaves like IRM — its
  AIPW-style score leaves fold-conditional construction details
  unspecified, so `sp.dml` and `doubleml-for-py` agree to ~1.2 × 10⁻²
  (≈ 0.13 SE) rather than to machine precision. Both land near the true
  LATE of 0.5 (0.549 vs 0.562). The external parity test tolerates 0.05
  absolute, matching the IRM discipline.

All four pins were re-verified on 2026-06-12 with StatsPAI 1.17.0
against `doubleml-for-py` 0.11.3 / scikit-learn 1.7.2 (4/4 passing,
PLR and PLIV at machine precision); the three score/IPW option pins
below were added against the same `doubleml-for-py` 0.11.3 (7/7
passing).

## DoubleML-compatible score and IPW options

`sp.dml` exposes the same score and inverse-propensity options as
DoubleML, with **defaults that reproduce the historical StatsPAI
estimate bit-for-bit** (so upgrading never moves a default result):

| Option | Models | Values (default **bold**) | DoubleML equivalent |
| --- | --- | --- | --- |
| `score` | `plr` | **`'partialling out'`**, `'IV-type'` | `DoubleMLPLR(score=...)` |
| `score` | `irm` | **`'ATE'`**, `'ATTE'` | `DoubleMLIRM(score=...)` |
| `normalize_ipw` | `irm`, `iivm` | **`False`**, `True` | `normalize_ipw=` |
| `trimming_threshold` | `irm`, `iivm` | **`0.01`** | `trimming_threshold=` (rule `'truncate'`) |

Each option is pinned against `doubleml-for-py` 0.11.3 in
[`tests/external_parity/test_dml_python_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/external_parity/test_dml_python_parity.py),
with same-seed agreement (the residual is fold-construction noise, the
same source as the IRM/IIVM pins):

| Option path | `sp.dml` vs `doubleml-for-py` |
| --- | --- |
| PLR `score='partialling out'` (default) | machine precision (\|Δ\| = 0) |
| PLR `score='IV-type'` | ≈ 0.011 (fold construction; not fold-invariant) |
| IRM `score='ATE'` (default) | ≈ 6 × 10⁻⁴ |
| IRM `score='ATTE'` | ≈ 0.011 |
| IRM / IIVM `normalize_ipw=True` | tracks DoubleML's shift (≤ 0.01) |

Two implementation notes worth stating for an auditor:

- **PLR `'IV-type'`** swaps the moment denominator from `Σ v̂²` to
  `Σ v̂·D` (`v̂ = D − m̂(X)`), reusing the same `E[Y|X]` outcome
  nuisance — no extra learner. Unlike partialling-out it is *not*
  fold-invariant, hence the ≈ 0.011 fold gap rather than machine
  precision.
- **`normalize_ipw`** mirrors DoubleML's `_normalize_ipw` exactly,
  including the (initially surprising) detail that DoubleML's **IIVM**
  self-normalizes using the *treatment* `D` as the indicator — not the
  instrument `Z` — even though the propensity being normalized is
  `m̂ = P(Z=1|X)`. StatsPAI reproduces that convention for parity.
- **`score='ATTE'` with `sample_weight`** is rejected loudly: DoubleML's
  `weights` object is a GATE construct, not a survey weight, so the two
  are not silently combined.

## When to expect divergence

`sp.dml` deviates from `doubleml-for-py` only in implementation
details that the original Chernozhukov et al. (2018) score leaves
unspecified:

1. **Propensity trimming**: `sp.dml` clips propensities to
   `[0.01, 0.99]`; `doubleml-for-py`'s IRM/IIVM default is *identical* —
   `trimming_rule='truncate'`, `trimming_threshold=0.01` — so the clip is
   *not* a source of divergence (and both are now adjustable via
   `trimming_threshold=`). On this fixture few propensities approach the
   boundary, so the clip is *not* what drives the small IRM gap. Trimming
   matters only when the estimated propensity has mass near 0 or 1, where
   the AIPW score is numerically unstable.
2. **Repeated cross-fitting aggregation**: `n_rep > 1` aggregates by
   median in both. With `n_rep=1` the seed fully determines folds.
3. **Convenience defaults**: `sp.dml`'s string aliases
   (`ml_g='rf'`, etc.) map to specific scikit-learn configurations
   (e.g. `RandomForestRegressor(n_estimators=200)`). Passing an
   explicit sklearn estimator removes this layer.

For audit-grade numerical equivalence, supply the same
`sklearn`-compatible estimators to both libraries (as the external
parity test does): the partialling-out models (PLR, PLIV) then agree
with `doubleml-for-py` to machine precision under a fixed seed
(verified above), and the AIPW models (IRM, IIVM) agree up to the small
score-construction difference noted above (≈ 0.10–0.13 SE). All four
DoubleML model classes are pinned numerically against `doubleml-for-py`
in `tests/external_parity/test_dml_python_parity.py`.

## Estimation procedure and nuisance learners

**DML2, not DML1.** `sp.dml` solves the *pooled* moment equation across
all cross-fitting folds at once — i.e. the **DML2** estimator of
Chernozhukov et al. (2018, Def. 3.2), which is also DoubleML's default
`dml_procedure`. For the partially-linear models this is the closed-form
`theta = sum(d_tilde * y_tilde) / sum(d_tilde**2)` over *all* out-of-fold
residuals (not a per-fold DML1 average). This pooled identity, the
solved Neyman moment, and the sandwich-variance formula are checked
directly in `tests/test_dml_orthogonality_invariants.py`. The per-fold
DML1 procedure is not currently exposed; for well-sized folds the two
agree closely and DML2 is the recommended default.

**Nuisance learners.** Any scikit-learn-compatible estimator can be
passed to `ml_g` / `ml_m` / `ml_r`, exactly as in DoubleML; string
aliases (`'lasso'`, `'rf'`, `'gbm'`, …) are convenience shortcuts for
common configurations. The scikit-learn cross-validated `LassoCV` is
the default sparse-linear nuisance, and any custom estimator with a
`fit`/`predict` API can be supplied.

**Rigorous / plug-in lasso (`hdm`).** The DML rate conditions are
stated for a theory-driven *plug-in* (rigorous) penalty rather than a
cross-validated one; the canonical implementation is the `rlasso` /
`rlassoIV` family in the R `hdm` package (Chernozhukov, Hansen &
Spindler, *The R Journal* 8(2), 2016 [@chernozhukov2016hdm]). StatsPAI
implements the Belloni–Chernozhukov–Hansen rigorous-penalty machinery
via the dedicated `sp.rlasso` module — a **faithful, parity-tested port
of `hdm`** (`hdm::rlasso` / `rlassoEffect` / `rlassoIV`). As of this
release:

- `sp.rlasso(X, y)` matches `hdm::rlasso` to **machine precision**
  (coefficients, `λ₀`, loadings, residuals and selected support are
  bit-exact across `post`/`intercept`/`homoscedastic` variants);
- `sp.rlasso_iv(y, d, z, x, select_Z=, select_X=)` reproduces
  `hdm::rlassoIV` on the BCH eminent-domain application **exactly**
  (`coef 0.2274`, `SE 0.2466`), across all four selection regimes;
- `sp.dml(model='plr', ml_g='rlasso', ml_m='rlasso')` now wires the
  rigorous Lasso in as a drop-in nuisance learner
  (`sp.RlassoRegressor` / `sp.RlassoClassifier`, clone-safe across
  folds).

This **resolves** the previously-tracked divergence: the earlier
hand-rolled reconstruction (`iv.bch_post_lasso_iv`, with the asymptotic
penalty `λ = 2c√{2n log(2p/α)}`) was ~17× off `hdm` on eminent domain
(0.013 vs 0.227) because it used a different penalty and selected only
instruments. `iv.bch_post_lasso_iv` retains its original numerics for
backward compatibility, but `sp.rlasso_iv` is the hdm-faithful path; see
[`docs/guides/rigorous_lasso_hdm.md`](rigorous_lasso_hdm.md). `LassoCV`
(CV-tuned) remains available as the alternative sparse-linear nuisance.

## Scope and known limitations

These are deliberate scope boundaries, declared up front so they are
discovered here rather than at runtime. Each one fails loudly (a
`ValueError` / `NotImplementedError` with a workaround in the message),
never silently.

1. **One scalar instrument per call (PLIV, IIVM).** The cross-fit
   reduced form `r(X) = E[Z|X]` is a scalar learner output, so
   `instrument=` accepts a single column. Passing a list of several
   instruments raises a `ValueError` that points to the escape hatch:
   `sp.scalar_iv_projection(data, treat=..., instruments=[...],
   covariates=...)` builds an OLS first-stage scalar index column you
   can pass as the instrument. Native vector-`Z` support (stacked
   moment + Cragg–Donald style first-stage diagnostics) is a tracked
   deferral — see `docs/ROADMAP.md` item 2.
2. **One treatment column per call.** `doubleml-for-py` accepts
   multiple `d_cols` on one `DoubleMLData` and offers multiplier-
   bootstrap simultaneous confidence bands across them; `sp.dml`
   estimates one treatment per call. For several treatments, call
   `sp.dml` per treatment and adjust with `sp.romano_wolf` /
   `sp.adjust_pvalues` if simultaneous inference is needed.
3. **Named scores, not arbitrary callables.** Both PLR scores
   (`'partialling out'`, `'IV-type'`) and both IRM scores (`'ATE'`,
   `'ATTE'`) are exposed via `score=` (see *DoubleML-compatible score
   and IPW options* above). DoubleML additionally accepts a fully custom
   callable score; `sp.dml` exposes the named scores, and a bespoke
   orthogonal score is added by subclassing `_DoubleMLBase` (see
   *Extending sp.dml with a custom score* below).
4. **DML2 only.** The pooled-moment DML2 procedure (both libraries'
   default) is the only one implemented; per-fold DML1 is not exposed
   (see [Estimation procedure](#estimation-procedure-and-nuisance-learners)).
5. **`fold_indices` is PLR-only.** Explicit fold assignments (for
   audit-grade fold reproduction) currently work for `model='plr'`;
   other models raise `NotImplementedError`.
6. **Cluster-robust DML standard errors are panel-only.** The four
   cross-sectional models report the heteroskedasticity-robust sandwich
   variance of the orthogonal score, as in Chernozhukov et al. (2018).
   `doubleml-for-py` additionally supports clustered data via
   `DoubleMLClusterData`; in StatsPAI, cluster-robust DML inference is
   available through `sp.dml_panel` (unit-level Liang–Zeger SEs for
   panel PLR with fixed effects), not through `sp.dml` itself.

One earlier limitation is **resolved** and worth stating explicitly
because older notes still mention it: `sample_weight=` (survey /
probability weights) is supported for **all four** models
(`plr`, `irm`, `pliv`, `iivm`), with weighted nuisance fits and a
weighted-moment sandwich variance.

## Companion tooling around the shared core

Beyond the four estimator classes that mirror DoubleML one-to-one,
`sp.dml` results plug into StatsPAI-side tooling:

- **`sp.dml_sensitivity(result)`** — omitted-variable-bias bounds and
  robustness values (`RV_q`, `RV_{q,α}`) for PLR / IRM estimates,
  following Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis
  [@chernozhukov2022long], with a `sensemakr`-style interface
  [@cinelli2020making] built on the DML residuals.
- **`sp.dml_diagnostics(result)`** — a four-panel report (propensity /
  residual overlap, orthogonal-score density, post-residualisation
  covariate balance, orthogonality test) with a publication-ready
  `.plot()`.
- **`sp.dml_panel(...)`** — long-panel DML with unit (and optional
  time) fixed effects and unit-clustered SEs, after Clarke & Polselli
  [@clarke2025double].
- **`sp.dml_model_averaging(...)`** — short-stacking / stacking over
  candidate nuisance learners, after Ahrens, Hansen, Schaffer & Wiemann
  [@ahrens2025model].
- **Unified result object** — every `sp.dml` call returns a
  `CausalResult`, so `.summary()`, `.to_latex()`, `.cite()` and the
  agent-side audit chain (`sp.audit`) work the same as for every
  other StatsPAI estimator.

## Extending `sp.dml` with a custom score

DoubleML's object-oriented design lets users plug in a custom callable
score. StatsPAI's analogue is subclassing `_DoubleMLBase`: set the model
metadata as class attributes and implement `_fit_one_rep`, which returns
`(theta, se)` for one cross-fitting repetition. The base class handles
data validation, learner resolution, repeated-cross-fit (median)
aggregation, and the `CausalResult` wrapping — so a new orthogonal score
is ~20 lines:

```python
import numpy as np
from sklearn.model_selection import KFold
from statspai.dml._base import _DoubleMLBase

class MyPLR(_DoubleMLBase):
    """Partially-linear DML with a custom Neyman-orthogonal score."""
    _MODEL_TAG = "MYPLR"
    _ESTIMAND = "ATE"

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed,
                     sample_weight=None, fold_indices=None):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
        y_res, d_res = np.zeros(n), np.zeros(n)
        for tr, te in kf.split(X):
            g = self._fit_weighted(self.ml_g, X[tr], Y[tr], None)
            m = self._fit_weighted(self.ml_m, X[tr], D[tr], None)
            y_res[te] = Y[te] - g.predict(X[te])
            d_res[te] = D[te] - m.predict(X[te])
        theta = float(np.sum(d_res * y_res) / np.sum(d_res**2))
        psi = (y_res - theta * d_res) * d_res
        J = -np.mean(d_res**2)
        se = float(np.sqrt(np.mean(psi**2) / (J**2 * n)))
        return theta, se

est = MyPLR(df, y="y", treat="d", covariates=cols,
            ml_g="linear", ml_m="linear", n_folds=5)
result = est.fit()          # -> a standard CausalResult
```

The four shipped models (`plr` / `irm` / `pliv` / `iivm`) are exactly
such subclasses; reading `src/statspai/dml/plr.py` is the recommended
template.

## Roadmap

Items where `doubleml-for-py` is broader and StatsPAI alignment is
tracked but not yet shipped:

- **Multiplier-bootstrap simultaneous inference** across several
  treatments (DoubleML's `bootstrap()` + joint `confint`). Today, run
  `sp.dml` per treatment and combine with `sp.romano_wolf` /
  `sp.adjust_pvalues`.
- **Multiway cluster-robust cross-fitting** (Chiang, Kato, Ma & Sasaki,
  *JBES* 40(3), 2022 [@chiang2022multiway]). Cross-sectional `sp.dml` reports the
  heteroskedastic sandwich SE; unit-clustered panel DML is available via
  `sp.dml_panel`.
- **Nested-CV learner tuning** (DoubleML's `.tune()`). Today, pass a
  pre-tuned scikit-learn `Pipeline` / `GridSearchCV` estimator.
- ~~**Rigorous-lasso (`hdm`) nuisance learner**~~ — **done**: `sp.rlasso`
  ports `hdm` to machine precision and `ml_g='rlasso'` / `ml_m='rlasso'`
  are validated nuisance aliases (see *Rigorous / plug-in lasso* above
  and [`docs/guides/rigorous_lasso_hdm.md`](rigorous_lasso_hdm.md)).
- **Automatic debiased ML / Riesz representers** (Chernozhukov, Newey &
  Singh) — a forward direction shared with the DoubleML ecosystem.

## Running the parity tests yourself

```bash
pip install -e ".[dev,parity]"   # the parity extra adds doubleml-for-py
                                  # (not a runtime dependency of StatsPAI);
                                  # pins were last verified against
                                  # doubleml-for-py 0.11.3.

# Python-side parity (sp.dml vs doubleml-for-py): 7 pins — 4 core models
# plus the score/IPW options (PLR IV-type, IRM ATTE, normalize_ipw). The
# partialling-out models match to machine precision; AIPW / IV-type paths
# match within the documented fold-construction tolerance.
# Without the parity extra this test skips cleanly instead of failing.
pytest tests/external_parity/test_dml_python_parity.py -v

# R-side parity (requires R + DoubleML + mlr3 installed locally)
pytest tests/reference_parity/test_dml_parity.py -v
```

The R-side fixture (`dml_R.json`) was generated once on R 4.5.2 with
`DoubleML` 1.0.2 + `mlr3learners` 0.14.0 + `cv_glmnet`; rerun
`_generate_dml.R` only when the DGP itself changes.

## References

- **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
  Hansen, C., Newey, W. & Robins, J. (2018).** Double/debiased
  machine learning for treatment and structural parameters. *The
  Econometrics Journal*, 21(1), C1–C68. [@chernozhukov2018double]
- **Bach, P., Chernozhukov, V., Kurz, M.S. & Spindler, M. (2022).**
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in Python. *Journal of Machine Learning Research*,
  23(53), 1–6. [@bach2022doubleml]
- **Bach, P., Kurz, M.S., Chernozhukov, V., Spindler, M. & Klaassen,
  S. (2024).** DoubleML — An Object-Oriented Implementation of
  Double Machine Learning in R. *Journal of Statistical Software*,
  108(3), 1–56. DOI: [`10.18637/jss.v108.i03`](https://doi.org/10.18637/jss.v108.i03). [@bach2024doubleml]
