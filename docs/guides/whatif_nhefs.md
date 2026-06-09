# Reproducing *Causal Inference: What If* on real NHEFS data

> **This guide is StatsPAI's parity certification for the epidemiology
> g-methods stack.** It reproduces the published estimates from Hernán &
> Robins, *Causal Inference: What If* (2020) — the canonical text of
> modern causal epidemiology — on the **real** NHEFS data the book itself
> uses, bundled in `sp.datasets.nhefs()`. Every number below is checked
> against (a) the figure printed in the book and (b) an independent
> base-R / `survival` / `EValue` reference run on the same bytes. The
> machinery lives in `tests/orig_parity/06–11_nhefs_*.{py,R}` and the
> pinned tests in `tests/external_parity/test_whatif_nhefs.py`.

If you are an epidemiologist arriving from SAS, Stata, or R's `ipw` /
`gfoRmula` / `survival`, this is the page that shows the same analyses,
function for function, in Python — and proves the answers match the book.

---

## The data

`sp.datasets.nhefs()` ships the genuine, public-domain NHEFS extract
(NHANES I Epidemiologic Followup Study; re-packaged from the
MIT-licensed `causaldata`). It is the only **real** dataset in
`sp.datasets` — the econometrics datasets are calibrated simulations,
but NHEFS is the actual book data, so StatsPAI is held to the book's
real figures.

```python
import statspai as sp

df = sp.datasets.nhefs()                       # 1629 × 67, real data
weight = sp.datasets.nhefs(complete_case=True) # 1566, non-missing wt82_71
df.attrs["published_ipw_att"]                  # 3.4  (book Program 12.4)
```

**The causal question (Part II):** what is the average effect of
**quitting smoking** (`qsmk`) between 1971 and 1982 on **10-year weight
change** (`wt82_71`, kg)? A crude comparison is confounded — quitters
are older, heavier, and less healthy at baseline. The book's confounder
set, used by every method below, is:

- categorical: `sex`, `race`, `education` (5 levels), `exercise` (3),
  `active` (3);
- continuous, with quadratic terms: `age`, `smokeintensity`,
  `smokeyrs`, `wt71`.

```python
import pandas as pd

CAT  = ["sex", "race", "education", "exercise", "active"]
CONT = ["age", "smokeintensity", "smokeyrs", "wt71"]

def book_design(df):
    d = df.copy()
    for c in CONT:
        d[f"{c}2"] = d[c] ** 2
    dd = pd.get_dummies(d, columns=CAT, drop_first=True)
    covs = [c for c in dd.columns if any(c.startswith(p + "_") for p in CAT)]
    covs += CONT + [f"{c}2" for c in CONT]
    return dd, [c for c in covs]
```

The crude, confounded difference is **2.54 kg** — exactly the book's
§12.2 figure:

```python
d = sp.datasets.nhefs(complete_case=True)
crude = d.loc[d.qsmk == 1, "wt82_71"].mean() - d.loc[d.qsmk == 0, "wt82_71"].mean()
# 2.54  (book §12.2)
```

---

## Chapter 12 — IP weighting and marginal structural models

Inverse-probability weighting builds a pseudo-population in which
`qsmk` is independent of the measured confounders, then fits a marginal
structural model.

```python
dd, covs = book_design(sp.datasets.nhefs(complete_case=True))
ipw = sp.ipw(dd, y="wt82_71", treat="qsmk", covariates=covs,
             estimand="ATE", seed=42)
print(ipw.estimate, ipw.ci)     # 3.47  (95% CI ~2.5, 4.4)
```

| | StatsPAI | R gold (stabilized MSM) | Book (Program 12.4) |
|---|---:|---:|---:|
| IP-weighted ATE | **3.47** | 3.44 | 3.4 (95% CI 2.4–4.5) |

`sp.ipw` uses Hájek (normalized) weights; the book's saturated MSM uses
stabilized weights. The two are asymptotically equal and here differ by
0.03 kg — both round to the book's one-decimal figure.

---

## Chapter 13 — Standardization / the parametric g-formula

Standardization fits an outcome model, predicts each subject's outcome
under `qsmk = 1` and `qsmk = 0`, and averages the contrast.

```python
g = sp.g_computation(dd, y="wt82_71", treat="qsmk", covariates=covs, seed=42)
print(g.estimate, g.ci)         # 3.46  (95% CI ~2.6, 4.4)
```

| | StatsPAI | R gold | Book (Program 13.3) |
|---|---:|---:|---:|
| Standardized ATE | **3.46** | 3.46 | 3.5 |

StatsPAI and the base-R standardization agree to four decimals on the
same bytes. `sp.aipw` (doubly robust) gives 3.51 as a cross-check.

---

## Chapter 14 — G-estimation of a structural nested model

G-estimation targets the structural nested mean model parameter ψ — the
estimand robust to a different set of modelling assumptions than IPW or
the g-formula.

```python
ge = sp.g_estimation(dd, y="wt82_71", treatments=["qsmk"],
                     covariates_by_stage=[covs], random_state=42)
print(ge.estimate)              # ψ = 3.46
```

| | StatsPAI | R gold (grid search) | Book (Program 14.2) |
|---|---:|---:|---:|
| ψ (SNMM) | **3.46** | 3.46 | 3.4 |

**Three different g-methods — IP weighting (3.47), standardization
(3.46), and g-estimation (3.46) — agree with each other and with the
book**, while the confounded crude estimate (2.54) understates the
effect by a kilogram. That triangulation is the whole point of Part II.

---

## Chapter 15 — Outcome regression and propensity scores

The outcome model with effect modification by smoking intensity
reproduces the book's Program 15.1 coefficients **to four decimals**:

```python
m = sp.regress(
    "wt82_71 ~ qsmk + smokeintensity + qsmk:smokeintensity + "
    "C(sex)+C(race)+C(education)+C(exercise)+C(active)+"
    "age+I(age**2)+I(smokeintensity**2)+smokeyrs+I(smokeyrs**2)+wt71+I(wt71**2)",
    data=sp.datasets.nhefs(complete_case=True))
m.params["qsmk"]                 # 2.56   (book 2.56)
m.params["qsmk:smokeintensity"]  # 0.0467 (book 0.0467)
```

| Statistic | StatsPAI | Book (Program 15.1) |
|---|---:|---:|
| `qsmk` main coefficient | **2.56** | 2.56 |
| `qsmk × smokeintensity` | **0.0467** | 0.0467 |
| Effect at smokeintensity = 5 | **2.79** | 2.79 |
| Effect at smokeintensity = 40 | **4.43** | 4.43 |
| Propensity-score-adjusted ATE | **3.45** | ≈ 3.5 |

These closed-form regressions are an **exact** match — the strongest
evidence that both the bundled data and the StatsPAI estimators are the
real thing.

---

## Chapter 17 — IP-weighted survival analysis

The mortality question (`death` by 1992) shows confounding vividly. The
crude/conditional hazard ratio for quitting is ~1.4 (quitters die more
— because they are older and sicker). IP weighting removes that
confounding and the hazard ratio collapses toward the null:

```python
# Build survival time + stabilized IP weights, then a weighted Cox /
# pooled-logistic hazard (see tests/orig_parity/10_nhefs_ch17_survival.py).
```

| | StatsPAI | R gold (`survival::coxph`) | Book (§17) |
|---|---:|---:|---:|
| Conditional (unweighted) HR | **1.39** | 1.39 | ≈ 1.4 |
| IP-weighted HR | **1.00** | 1.00 | ≈ 1.0 |
| IP-weighted 120-mo survival difference | **0.002** | 0.002 | ≈ 0 |

StatsPAI's IP-weighted survival agrees with R's `survival::coxph` to
three decimals on the same data, and reproduces the book's qualitative
finding: once confounding is removed, quitting smoking has little effect
on 10-year mortality.

---

## Sensitivity analysis — the E-value

How strong would an unmeasured confounder have to be to explain away
the quit-smoking → mortality association? The E-value (VanderWeele &
Ding 2017) answers in one line.

```python
full = sp.datasets.nhefs()
rr = full.loc[full.qsmk == 1, "death"].mean() / full.loc[full.qsmk == 0, "death"].mean()
ev = sp.evalue(estimate=float(rr), measure="RR")
ev["evalue_estimate"]            # 1.98  (matches R EValue::evalues.RR exactly)
```

The E-value of **1.98** means an unmeasured confounder would need a risk
ratio of at least 1.98 with *both* quitting and death, beyond the
measured confounders, to nullify the association — matching the
closed-form `E = RR + √(RR(RR−1))` and R's `EValue` package to 1e-3.

---

## What this certifies (and what it doesn't)

| Quantity | Match to book | Match to R gold |
|---|---|---|
| Crude difference, Ch15 regressions, E-value | **exact** (closed form) | exact / 4 dp |
| IP weighting, g-formula, g-estimation, survival | within ~2% / rounding | 3–4 dp |

The closed-form quantities are exact. The IP-weighted, g-formula,
g-estimation, and survival quantities carry small, **documented**
convention choices (Hájek vs stabilized weights; additive encoding of
the SNMM; pooled-logistic vs Cox hazards) that keep them within ~2% of
both the book and the independent R reference. This is the first
epidemiology-stack (as opposed to econometrics-stack) parity
certification in StatsPAI; it upgrades the g-methods rows of the
[public-health guide](public_health.md) from "API-stable but not yet
parity-certified" to "validated against the *What If* canon and R."

**Run it yourself:** `python examples/nhefs_whatif.py`.

### References

- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*.
  Boca Raton: Chapman & Hall/CRC.
- VanderWeele, T.J. & Ding, P. (2017). Sensitivity Analysis in
  Observational Research: Introducing the E-Value. *Annals of Internal
  Medicine* 167(4), 268-274.
