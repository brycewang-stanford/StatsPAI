# Decomposition Analysis

`statspai.decomposition` — 18 decomposition methods across 13 modules
(v0.9.2, ~6,200 LOC, 54 tests). The most comprehensive decomposition
toolkit in Python, covering mean, distributional, inequality,
demographic, and causal decomposition.

## Unified dispatcher — `sp.decompose`

```python
r = sp.decompose(
    method='ffl',                 # 30 aliases accepted
    data=df, y='log_wage', group='female',
    x=['education', 'experience'],
    stat='quantile', tau=0.5,
    inference='analytical',       # 'analytical' | 'bootstrap' | 'none'
)
r.summary(); r.plot(); r.to_latex()
```

## Mean decomposition

| Function | Method / Paper |
| --- | --- |
| `sp.oaxaca(df, ...)` | Blinder-Oaxaca threefold with 5 reference coefficients (Blinder 1973, Oaxaca 1973, Neumark 1988, Cotton 1988, Reimers 1983) |
| `sp.gelbach(df, ...)` | Sequential orthogonal decomposition (Gelbach 2016, JoLE) |
| `sp.fairlie(df, ...)` | Nonlinear logit/probit decomposition (Fairlie 1999, 2005) |
| `sp.bauer_sinning(df, ...)` / `sp.yun_nonlinear(df, ...)` | Detailed nonlinear (Bauer-Sinning 2008; Yun 2004/05) |

## Distributional decomposition

| Function | Method / Paper |
| --- | --- |
| `sp.rifreg(df, ...)` / `sp.rif_decomposition(...)` | RIF regression + OB (Firpo-Fortin-Lemieux 2009, *Econometrica*) |
| `sp.ffl_decompose(df, ...)` | Two-step detailed (FFL 2018) |
| `sp.dfl_decompose(df, ...)` | Reweighting counterfactuals (DiNardo-Fortin-Lemieux 1996) |
| `sp.machado_mata(df, ...)` | Simulation-based QR decomposition (MM 2005) |
| `sp.melly_decompose(df, ...)` | Analytical QR decomposition (Melly 2005) |
| `sp.cfm_decompose(df, ...)` | Distribution regression (Chernozhukov-Fernández-Val-Melly 2013) |

## Inequality decomposition

| Function | Method / Paper |
| --- | --- |
| `sp.subgroup_decompose(df, ...)` | Between/within for Theil T/L, GE(α), Dagum Gini (1997), Atkinson, CV² (Shorrocks 1984) |
| `sp.shapley_inequality(df, ...)` | Shorrocks-Shapley allocation to covariates (Shorrocks 2013) |
| `sp.source_decompose(df, ...)` | Gini source decomposition (Lerman-Yitzhaki 1985) |

## Demographic standardisation

| Function | Method / Paper |
| --- | --- |
| `sp.kitagawa_decompose(df, ...)` | Two-factor rate decomposition (Kitagawa 1955) |
| `sp.das_gupta(df_a, df_b, ...)` | Multi-factor symmetric (Das Gupta 1993) |

## Causal decomposition

| Function | Method / Paper |
| --- | --- |
| `sp.gap_closing(df, method=...)` | Gap-closing estimator (Lundberg 2021), regression / IPW / AIPW |
| `sp.mediation_decompose(df, ...)` | Natural direct/indirect effects (VanderWeele 2014) |
| `sp.disparity_decompose(df, ...)` | Causal disparity decomposition (Jackson-VanderWeele 2018) |

## Quality bar

- Closed-form influence functions for Theil T / Theil L / Atkinson
  (no O(n²) numerical fallback).
- Weighted O(n log n) Dagum Gini via sorted-ECDF pairwise-MAD identity.
- Cross-method consistency tests: `test_dfl_ffl_mean_agree`,
  `test_mm_melly_cfm_aligned_reference`,
  `test_dfl_mm_reference_convention_opposite`.
- Numerical identity checks: FFL four-part sum, weighted Gini RIF
  $E_w[\text{RIF}] = G$.
