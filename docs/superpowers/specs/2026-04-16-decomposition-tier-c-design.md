# StatsPAI Decomposition Analysis — Tier C Design

**Date:** 2026-04-16
**Author:** Bryce Wang + Claude
**Status:** Approved, execution authorized
**Scope:** Tier C — full decomposition suite with causal view, inequality decomposition, Kitagawa standardization, and ecosystem integration.

---

## 1. Goal

Build the world's most powerful decomposition analysis toolkit in Python, covering mean, distributional, inequality, demographic, and causal decomposition under a unified API — `sp.decompose(method=...)`.

**Competitive target:** Beat Stata `ddecompose` (Rios-Avila 2024), Stata `cdeco`, R `Counterfactual`, R `ddecompose`, and occupy the empty Python high-ground in one pass.

---

## 2. Method Coverage (18 methods in 11 modules)

### 2.1 Mean decomposition (`oaxaca.py`, existing — enhance)

- Threefold Oaxaca-Blinder ✓ (exists)
- 5 reference coefficients: Group A, Group B, Pooled (Neumark), Cotton, Reimers ✓ (exists)
- **ADD**: Oaxaca-Ransom (1994) pooled with group dummy reference
- **ADD**: Yun (2005) normalization for categorical covariates (invariant to omitted category)
- **ADD**: Bootstrap SE (stratified, cluster-robust)
- **ADD**: Jann-style detailed both explained and unexplained sides
- **ADD**: Sample weights support
- Gelbach (2016) ✓ (exists, keep)

### 2.2 RIF regression & decomposition (`rif.py`, existing — enhance)

- RIF values for quantile / variance / Gini ✓ (exists)
- **ADD**: RIF for IQR, Atkinson index, Theil index, log-variance
- **ADD**: Detailed coefficient-side contributions (not just explained)
- **ADD**: Multiple reference groups (0, 1, pooled)
- **ADD**: Bootstrap SE
- **ADD**: Reweighting adjustment (FFL two-step)

### 2.3 FFL detailed distributional decomposition (`ffl.py`, NEW)

- Firpo-Fortin-Lemieux (2009, 2018) two-step:
  1. Reweighting (DFL) to isolate composition effect
  2. RIF regression on reweighted sample for structural effect
- Detailed breakdown at any τ, variance, Gini, etc.
- Aggregation across quantiles → quantile process

### 2.4 DFL reweighting (`dfl.py`, NEW)

- DiNardo, Fortin, Lemieux (1996) reweighting of distributions
- Logit/probit propensity score for weights
- Counterfactual density, quantile, CDF
- Support for Hirano-Imbens-Ridder stabilized weights
- Optional entropy balancing weights (Hainmueller 2012) as alternative
- Full distributional pointwise + uniform confidence bands (bootstrap)

### 2.5 Machado-Mata quantile decomposition (`machado_mata.py`, NEW)

- Machado & Mata (2005): quantile regression + resampling
- Detailed decomposition at each τ ∈ (0,1)
- Optional Albrecht-Björklund-Vroman extension
- Bootstrap inference

### 2.6 Melly simulation-based decomposition (`melly.py`, NEW)

- Melly (2005, 2006): integrate QR coefficients over τ
- Faster and more consistent than Machado-Mata
- Both conditional and unconditional decomposition

### 2.7 Chernozhukov-Fernández-Val-Melly counterfactual distributions (`cfm.py`, NEW)

- CFM 2013: distribution regression / quantile regression / duration regression
- Counterfactual CDF F_{Y<j|k>}
- Quantile function Q_{Y<j|k>}
- Uniform confidence bands via empirical bootstrap + Khmaladze transform (optional)
- KS, CvM tests for distributional equality
- Detailed decomposition at any functional (Lorenz, GL, variance, quantile ratios)

### 2.8 Nonlinear decomposition (`nonlinear.py`, NEW)

- Fairlie (1999, 2005) for binary outcomes (logit/probit)
- Bauer-Sinning (2008, 2010) extension
- Yun (2005) nonlinear Oaxaca
- Powers-Yun (2011) multivariate nonlinear (mvdcmp equivalent)

### 2.9 Inequality decomposition (`inequality.py`, NEW)

- **By subgroup** (between/within): Theil T, Theil L, Atkinson, MLD, half-squared CV, Gini
- **By source**: Lerman-Yitzhaki (1985) Gini source decomposition
- **Shapley** (Shorrocks 2013): allocate inequality to covariates via RIF + Shapley value
- Sample weights support
- Bootstrap inference

### 2.10 Kitagawa & Das Gupta standardization (`kitagawa.py`, NEW)

- Kitagawa (1955) direct standardization
- Das Gupta (1993) multivariate decomposition (rate × composition for multiple factors)
- Support for any aggregate (rate, mean, proportion)

### 2.11 Causal decomposition (`causal.py`, NEW)

- **Jackson-VanderWeele (2018)** causal decomposition:
  - Total disparity → initial + persistent (with mediator removed to reference level)
- **VanderWeele four-way** natural direct, natural indirect, mediated interaction, reference interaction
- **Lundberg (2021) gap-closing estimator**:
  - IPW, regression, doubly-robust (AIPW) versions
  - Closes mean gap under counterfactual covariate shift
- Sensitivity bounds (placeholder: wire to `sp.bounds` module)

### 2.12 Visualization (`plots.py`, NEW)

- **Waterfall chart** — detailed decomposition (mean and distributional)
- **Quantile process plot** — explained/unexplained vs τ with confidence bands
- **Counterfactual CDF / density** — overlay observed vs counterfactual
- **RIF heatmap** — variable × quantile contribution grid
- **Shapley treemap** — inequality decomposition
- All return `(fig, ax)` for user customization

### 2.13 Datasets (`datasets.py`, NEW)

- `cps_wage()` — CPS-like Mincer wage data (generated, realistic gap) [gender gap]
- `chilean_households()` — synthetic urban/rural income
- `mincer_wage_panel()` — pre/post wage structure shift
- `disparity_panel()` — synthetic panel for causal decomposition

### 2.14 Unified dispatcher (`dispatcher.py` + `__init__.py`)

```python
sp.decompose(
    method='oaxaca' | 'rif' | 'ffl' | 'dfl' | 'machado_mata' | 'melly' |
           'cfm' | 'fairlie' | 'bauer_sinning' | 'yun_nonlinear' |
           'inequality' | 'shapley_inequality' | 'kitagawa' |
           'das_gupta' | 'causal_jvw' | 'gap_closing' | 'gelbach',
    data=df, **method_kwargs,
)
```

Returns method-specific result class; all implement `.summary()`, `.plot()`, `.to_latex()`, `._repr_html_()`.

---

## 3. Architecture

### 3.1 File layout

```
src/statspai/decomposition/
├── __init__.py           # exports + unified dispatcher import
├── _common.py            # shared OLS/logit/bootstrap helpers
├── oaxaca.py             # [EXISTS] + Yun, Oaxaca-Ransom, bootstrap
├── rif.py                # [EXISTS] + IQR/Atkinson/Theil, detailed coeff, bootstrap
├── nonlinear.py          # NEW
├── dfl.py                # NEW
├── machado_mata.py       # NEW
├── melly.py              # NEW
├── cfm.py                # NEW
├── ffl.py                # NEW
├── inequality.py         # NEW
├── kitagawa.py           # NEW
├── causal.py             # NEW
├── datasets.py           # NEW
├── plots.py              # NEW
└── dispatcher.py         # NEW — sp.decompose()
```

### 3.2 Dependencies

- Core: numpy, scipy, pandas (already project standard)
- Optional: matplotlib (plots), statsmodels (for QR acceleration), sklearn (for propensity)
- All new modules degrade gracefully without optional deps

### 3.3 Result class contract

Every method returns a result object with:
- `.summary() → str` (also prints)
- `.plot() → (fig, ax)` (when meaningful)
- `.to_latex() → str`
- `._repr_html_() → str` (for Jupyter)
- `.__repr__() → str` (short one-liner)
- Tabular data accessible via `.table` or named attributes

### 3.4 Inference options (uniform)

- `inference='analytical'` — delta method / influence function (fast, default)
- `inference='bootstrap'` — stratified/cluster bootstrap, `n_boot` parameter
- `inference='none'` — point estimates only (fastest)

---

## 4. Backward compatibility

All existing public APIs preserved:
- `sp.oaxaca(...)` — unchanged signature, may accept new kwargs
- `sp.gelbach(...)` — unchanged
- `sp.rifreg(...)` — unchanged
- `sp.rif_decomposition(...)` — unchanged
- `OaxacaResult`, `GelbachResult`, `RIFResult`, `RIFDecompositionResult` — unchanged

New unified entry `sp.decompose(method='oaxaca', ...)` wraps the existing functions.

---

## 5. Test strategy

`tests/test_decomposition_tier_c.py` — one file covering:
- Each new method: smoke test (runs without error, right shape)
- Each new method: numerical sanity (against known closed form or simulation)
- API consistency: `.summary()` returns string, `.plot()` works when matplotlib available
- Unified dispatcher: each method reachable via `sp.decompose(method=...)`

Existing `tests/test_oaxaca*.py` / `tests/test_rif*.py` continue passing.

---

## 6. Commit plan

1. Spec doc (this file)
2. `_common.py` + enhancements to `oaxaca.py`, `rif.py`
3. `dfl.py` + `ffl.py`
4. `machado_mata.py` + `melly.py` + `cfm.py`
5. `nonlinear.py` + `inequality.py`
6. `kitagawa.py` + `causal.py`
7. `plots.py` + `datasets.py` + `dispatcher.py` + `__init__.py`
8. Tests + bug fixes
9. Final docs update + version bump

Direct push to `main` (per user preference).

---

## 7. Out of scope (for this sprint)

- Full Khmaladze test implementation for CFM (placeholder only; mark as future work)
- Dynamic treatment regimes decomposition (too speculative)
- Blundell-Dias-Meghir longitudinal decomposition (add later)
- Interactive dashboard (`sp.decompose_dashboard()`) — future
- R/Stata call-out (`compat/`) — future
