# Getting started — your first analysis in 5 minutes

This page takes you from `pip install` to a publication-ready difference-in-
differences estimate, with citation, in well under five minutes. Every code
block below is runnable as-is against a bundled dataset.

## 1. Install

```bash
pip install StatsPAI
```

That is all you need for the core estimators. Optional extras pull in heavier
backends only when you want them:

```bash
pip install "StatsPAI[plotting]"   # matplotlib / seaborn / plotly figures
pip install "StatsPAI[bayes]"      # PyMC + ArviZ for Bayesian estimators
pip install "StatsPAI[tune]"       # Optuna for tuned meta-learners / Auto-CATE
pip install "StatsPAI[rd-cct]"     # rdrobust for exact CCT RD parity
pip install "StatsPAI[performance]"# JAX backend for fast feols / bootstrap
```

## 2. One import

```python
import statspai as sp
```

Everything lives under `sp.` — there is no second-level import to remember.
`sp.list_functions()` enumerates all 1,000+ registered functions.

## 3. Load bundled data

StatsPAI ships the canonical teaching datasets so you can run real analyses
with zero setup:

```python
df = sp.datasets.mpdta()          # Callaway–Sant'Anna county teen employment
df.head()
#    countyreal  year      lemp  first_treat  treat
# 0           0  2003  8.162509         2004      0
# 1           0  2004  8.275744         2004      1
```

`sp.datasets.list_datasets()` shows the rest (Card 1995 schooling, Lee 2008
Senate RD, California Prop 99 synthetic control, LaLonde/NSW, …).

## 4. Let StatsPAI read the study design

Not sure which estimator fits? Ask:

```python
sp.detect_design(df)["design"]
# 'panel'
```

`sp.detect_design` inspects the data shape (cross-section / panel / RD / …)
and `sp.recommend(...)` suggests an estimator. This is the same machinery an
LLM agent uses to plan an analysis.

## 5. Estimate

`first_treat` is the year each county was first treated (the cohort), `year`
is time, `countyreal` is the unit, and `lemp` is log employment. That is a
staggered-adoption DiD, so use the heterogeneity-robust Callaway–Sant'Anna
estimator:

```python
r = sp.callaway_santanna(df, y="lemp", g="first_treat", t="year", i="countyreal")
print(r.summary())
# ==============================================================================
#   Callaway and Sant'Anna (2021)
# ==============================================================================
#   ATT:         -0.032977 ***
#   Std. Error:  (0.007740)
#   [95% CI]:    [-0.048146,  -0.017807]
#   P-value:      0.0
```

## 6. Check assumptions and sensitivity

```python
sp.agent_card("callaway_santanna")["assumptions"]
# ['Parallel trends conditional on X ...', 'No anticipation', 'SUTVA', ...]

sp.audit(r)            # which robustness checks are still missing?
sp.honest_did(r)       # Rambachan–Roth bounds on parallel-trends violations
```

## 7. Export for the paper

Every result object speaks the same export protocol:

```python
r.to_latex("att.tex")     # publication table
r.to_word("att.docx")
r.cite()                  # verified BibTeX for the estimator
```

## Where to next

- **[Cookbook](cookbook.md)** — recipes organised by research question.
- **[Choosing a DID estimator](guides/choosing_did_estimator.md)** and the
  other decision guides.
- **[FAQ](faq.md)** — common errors and how to read the diagnostics.
- **[Full API reference](reference/api/index.md)** — all 86 sub-packages.
