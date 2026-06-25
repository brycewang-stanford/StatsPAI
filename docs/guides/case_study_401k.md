# Case study: reproducing the DoubleML / hdm 401(k) result with `sp.dml`

The effect of **401(k) eligibility on net financial assets** is the
canonical applied example for double machine learning. It appears in the
[DoubleML][doubleml] "Getting Started" vignette and traces back to
Chernozhukov & Hansen (2004) and the `hdm` package (Chernozhukov, Hansen
& Spindler, *The R Journal* 8(2), 2016). This page shows that `sp.dml`
reproduces that result, side by side with `doubleml-for-py` on the *same
data*.

[doubleml]: https://github.com/DoubleML/doubleml-for-py

## Getting the data (no bundling)

StatsPAI ships **no copy** of the 401(k) data. The dataset is pulled
from DoubleML's own public distribution, so there is a single,
canonical, license-clear source and nothing to keep in sync:

```python
from doubleml.datasets import fetch_401K
df = fetch_401K(return_type="DataFrame")     # 9,915 households, 14 columns
# outcome: net_tfa (net financial assets)
# treatment: e401  (eligible for a 401(k))
# controls: age, inc, educ, fsize, marr, twoearn, db, pira, hown, nifa, tw
```

`fetch_401K` requires `doubleml` (`pip install doubleml`); it is **not**
a runtime dependency of StatsPAI. The extract originates from the 1991
SIPP (a U.S. Census survey) via Chernozhukov & Hansen's replication data.

## Estimating the effect with `sp.dml`

```python
import statspai as sp

y, d = "net_tfa", "e401"
covs = ["age", "inc", "educ", "fsize", "marr", "twoearn",
        "db", "pira", "hown", "nifa", "tw"]

# Partially linear model (lasso nuisances)
plr = sp.dml(data=df, y=y, treat=d, covariates=covs,
             model="plr", ml_g="lasso", ml_m="lasso",
             n_folds=5, random_state=42)
print(plr.summary())          # estimate ~ $8,200, se ~ $570

# Interactive (AIPW) model — ATE
irm = sp.dml(data=df, y=y, treat=d, covariates=covs,
             model="irm", ml_g="lasso", ml_m="logistic",
             n_folds=5, random_state=42)

# Rigorous-Lasso (hdm) nuisances — the theory-correct plug-in penalty
plr_hdm = sp.dml(data=df, y=y, treat=d, covariates=covs,
                 model="plr", ml_g="rlasso", ml_m="rlasso",
                 n_folds=5, random_state=42)
```

## Results — `sp.dml` vs `doubleml-for-py`

Run on the fetched data (StatsPAI 1.20.0, `doubleml-for-py` 0.11.3,
scikit-learn nuisances, `n_folds=5`, seed 42):

| Model | `sp.dml` estimate (se) | `doubleml-for-py` estimate (se) |
| --- | --- | --- |
| **PLR** (lasso) | **8 227.7** (566.3) | 8 166.9 (574.6) |
| **IRM** ATE (lasso + logistic) | 8 644.5 (1 742.6) | 8 156.5 (1 488.4) |
| **PLR**, rigorous-Lasso (`hdm`) nuisances | 8 302.8 (469.3) | — |

The `doubleml-for-py` numbers come from running its own
`DoubleMLPLR` / `DoubleMLIRM` on the identical `DataFrame`:

```python
import doubleml as dml
from sklearn.linear_model import LassoCV, LogisticRegressionCV

dd = dml.DoubleMLData(df, y_col=y, d_cols=d, x_cols=covs)
dml.DoubleMLPLR(dd, ml_l=LassoCV(), ml_m=LassoCV(), n_folds=5).fit()
dml.DoubleMLIRM(dd, ml_g=LassoCV(), ml_m=LogisticRegressionCV(), n_folds=5).fit()
```

**Reading the table.** All paths estimate a positive effect of roughly
**\$8,000–\$8,700** of net financial assets attributable to 401(k)
eligibility — the well-known magnitude reported throughout the DoubleML
and `hdm` literature. The PLR estimates agree to within \$60 (≈ 0.1 of a
standard error). The IRM estimates differ by ≈ \$490 — about **0.3 of a
standard error**, i.e. statistically indistinguishable — because the
AIPW score leaves fold-conditional construction details unspecified and
the two engines draw independent cross-fitting partitions here. The
rigorous-Lasso path (`ml_g='rlasso'`), which uses `hdm`'s theory-driven
plug-in penalty instead of cross-validation, lands in the same place.

## Why the small differences — and where parity is exact

These residuals are **fold randomization**, not a method discrepancy.
When `sp.dml` and `doubleml-for-py` are given the *same* scikit-learn
learners on the *same* fold partition under a fixed seed, the
partialling-out models (PLR, PLIV) agree to **machine precision**
(|Δ| at the last float64 unit), and the AIPW models (IRM, IIVM) agree up
to the small score-construction term. That bit-for-bit equivalence is
pinned offline (no network) in
[`tests/external_parity/test_dml_python_parity.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/external_parity/test_dml_python_parity.py)
and explained in detail in
[`sp.dml` and the DoubleML reference implementation](sp_dml_vs_doubleml.md).

For the rigorous-Lasso (`hdm`) side of the same workflow — `sp.rlasso`,
`rlasso_effect`, `rlasso_iv`, `rlassologit`, all pinned to `hdm` to
machine precision — see
[Rigorous (data-driven) Lasso — `sp.rlasso` and the `hdm` port](rigorous_lasso_hdm.md).

## References

- Chernozhukov, V. & Hansen, C. (2004). The Effects of 401(k)
  Participation on the Wealth Distribution: An Instrumental Quantile
  Regression Analysis. *Review of Economics and Statistics*, 86(3),
  735–751. doi
  [`10.1162/0034653041811734`](https://doi.org/10.1162/0034653041811734).
- Chernozhukov, V., Hansen, C. & Spindler, M. (2016). hdm:
  High-Dimensional Metrics. *The R Journal*, 8(2), 185–199.
  doi [`10.32614/RJ-2016-040`](https://doi.org/10.32614/RJ-2016-040).
- Bach, P., Chernozhukov, V., Kurz, M.S. & Spindler, M. (2022).
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in Python. *Journal of Machine Learning Research*, 23(53),
  1–6.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
  Newey, W. & Robins, J. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal*,
  21(1), C1–C68. doi
  [`10.1111/ectj.12097`](https://doi.org/10.1111/ectj.12097).
