# Panel Shift-Share IV for Political Science

> Park & Xu (arXiv:2603.00135, 2026) §4.2, with
> Adão-Kolesár-Morales (QJE 2019) variance.

## 1. When to use this

Use `sp.shift_share_political_panel` when **all** of the following hold:

- You have a **unit × time** panel (states × years, counties × election
  cycles, etc.).
- Your endogenous regressor is built from unit-specific *exposure
  shares* times a common *shift* vector — the canonical Bartik /
  shift-share setup.
- Either the shares **or** the shocks (or both) vary over time.
- You want pooled 2SLS with fixed effects plus the AKM
  shock-clustered SE that Park-Xu (2026) §4.2 identifies as the
  default for political-science panels.

For single-period (cross-section / long-difference) designs use
`sp.shift_share_political` or `sp.bartik` instead.

## 2. API

```python
import statspai as sp

res = sp.shift_share_political_panel(
    data=df,                    # long-format unit × time panel
    unit='state',
    time='year',
    outcome='vote_share',
    endog='exposure',

    # --- shares: either time-invariant (DataFrame unit×industry)
    # ---          or per-period (dict[time -> DataFrame])
    shares=shares,

    # --- shocks: either (1) Series (industry -> scalar; constant across time)
    # ---          or    (2) DataFrame (time × industry)
    # ---          or    (3) dict[time -> Series]
    shocks=shocks,

    covariates=['demographic_controls', 'fiscal_base'],
    fe='two-way',               # two-way | unit | time | none
    cluster='shock',            # shock | unit | time | twoway
    alpha=0.05,
)
print(res.summary())
```

Returns `ShiftSharePoliticalPanelResult` with

- `estimate`, `se`, `ci` — pooled 2SLS coefficient on `endog`
- `per_period` — DataFrame of cross-sectional estimates per `time`
  (turn it into an event-study plot directly)
- `rotemberg_panel` — industry-level weight decomposition, aggregated
  across all periods
- `share_balance` — F-test of pre-period covariates on the share matrix
- `diagnostics` — includes `cluster` label, `akm_se`, `first_stage_F`

## 3. The AKM shock-clustered SE

Standard unit- or two-way-clustered SEs are **inconsistent** for
shift-share designs: they ignore the fact that observations sharing a
shock are correlated through the common shifter, not through a geographic
cluster.  Adão-Kolesár-Morales (2019) derive the correct variance
estimator by clustering at the **shock** level with share-weighted
scores.  Park-Xu (2026) §4.2 extend it to the panel:

$$
u_k \;=\; \sum_{i, t}\, s_{ikt}\, \tilde Z_{it}\, \hat\varepsilon_{it},
\qquad
\widehat{\mathrm{Var}}(\hat\beta) \;=\;
\frac{\sum_k u_k^2}{\bigl(\hat D'_{\mathrm{fit}}\,\tilde D\bigr)^2}
$$

where $\tilde{\cdot}$ denotes FE-demeaning.  Pass `cluster='shock'` to
get this variance; the `diagnostics['akm_se']` field is also populated
so you can report both numbers side-by-side.

In clean Park-Xu-style DGPs with 10–100 industries the shock-clustered
SE is typically **3× tighter** than the unit-clustered SE, because it
correctly recognises that cross-unit correlation within a period is
driven by a few common shocks rather than by unit-level idiosyncrasies.

## 4. Fixed-effects choices

| `fe`         | What's absorbed           | When to pick                   |
|--------------|---------------------------|--------------------------------|
| `'two-way'`  | Unit + time FEs (default) | Almost always — Park-Xu default |
| `'unit'`     | Unit FEs only             | Time-invariant shares + strong time trend |
| `'time'`     | Time FEs only             | Balanced panel with few units  |
| `'none'`     | No FEs                    | You want pure cross-period pool |

FE demeaning is performed inside the function via mean-subtraction —
for >500 units × 20 periods consider pre-absorbing using
`sp.fixest.feols` then passing the residuals to this function with
`fe='none'`.

## 5. End-to-end recipe

```python
import numpy as np, pandas as pd, statspai as sp

rng = np.random.default_rng(0)
units, times, inds = list(range(100)), list(range(4)), [f'I{k}' for k in range(15)]

shares = pd.DataFrame(
    rng.dirichlet(np.ones(15), size=len(units)),
    index=units, columns=inds,
)
shocks = pd.DataFrame(
    rng.normal(size=(len(times), 15)),
    index=times, columns=inds,
)

rows = []
tau = 0.30
for i in units:
    for t in times:
        b = float((shares.loc[i] * shocks.loc[t]).sum())
        x = b + rng.normal(scale=0.1)
        y = tau * x + rng.normal(scale=0.1)
        rows.append({'u': i, 't': t, 'y': y, 'x': x})
df = pd.DataFrame(rows)

res = sp.shift_share_political_panel(
    df, unit='u', time='t', outcome='y', endog='x',
    shares=shares, shocks=shocks, cluster='shock',
)
print(res.summary())
```

Expected output on this DGP: estimate ≈ 0.296, shock-clustered
SE ≈ 0.009, per-period event-study stable around 0.3, the top-5
industries sum to ~95% of Rotemberg weight.

## 6. Pretrend / placebo checks

Because the per-period table `res.per_period` gives one estimate per
year, you can feed it straight into the event-study plotting helpers
or write your own pretrend Wald test:

```python
pre = res.per_period[res.per_period['time'] < treatment_year]
wald = (pre['estimate'] / pre['se']).pow(2).sum()
import scipy.stats as stats
p = 1 - stats.chi2.cdf(wald, df=len(pre))
print(f'Joint pretrend p-value: {p:.3f}')
```

## 7. Common pitfalls

- **Mis-aligned industry sets.**  If `shares.columns` and
  `shocks.index` don't share any labels the function raises.  When
  they only partially overlap the function silently intersects — use
  `res.n_industries` to confirm the size of the intersection.
- **Very-weak first stage.**  Check `res.diagnostics['first_stage_F']`;
  the shock-cluster variance estimator blows up when the FE-demeaned
  instrument is near-zero.
- **Share balance.**  A significant share-balance F on a pre-treatment
  covariate is a warning that the shares encode latent unit
  characteristics — consider adding the offending variable to
  `covariates=` and re-running.

## 8. References

- Park, P. K. & Xu, Y. (2026).
  *Shift-Share Designs in Political Science.*  arXiv:2603.00135.
- Adão, R., Kolesár, M. & Morales, E. (2019).
  *Shift-Share Designs: Theory and Inference.*  QJE 134(4).
- Borusyak, K., Hull, P. & Jaravel, X. (2022).
  *Quasi-Experimental Shift-Share Research Designs.*  ReStud 89.
