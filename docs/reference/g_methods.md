# G-methods family

Robins' **g-methods** family of causal-inference estimators, adapted
to the StatsPAI `CausalResult` API. These methods all share the
conceptual goal of evaluating counterfactual outcome distributions
under hypothetical interventions, but differ in which nuisance is
estimated (outcome model, treatment model, or both) and which
identifying assumption is leveraged.

| Estimator | When to reach for it | Identification |
| --- | --- | --- |
| `sp.g_computation` | Outcome model is trusted. Clean dose-response curves. | Unconfoundedness |
| `sp.front_door` | Unobserved back-door confounder, but a mediator fully transmits D's effect on Y. | Pearl's front-door criterion |
| `sp.msm` | Time-varying treatment with time-varying confounders affected by past treatment. | Sequential exchangeability + positivity |
| `sp.mediate_interventional` | Mediation with treatment-induced mediator-outcome confounders. | Interventional effects (no cross-world) |

Related modules (covered elsewhere):

* `sp.aipw`, `sp.tmle`, `sp.dml` — doubly-robust cousins of
  `g_computation` with orthogonal / efficient-influence-function
  inference.
* `sp.mediate` — classical natural (in)direct effects (Imai-Keele-
  Tingley 2010); see the note in the mediation section below.

---

## `sp.g_computation` — parametric g-formula / standardisation

Given an outcome regression :math:`Q(d, X) = E[Y | D=d, X]`, the
g-formula marginal counterfactual is

$$
E[Y(d)] = E_X[\, Q(d, X) \,].
$$

```python
# Binary treatment — ATE
r = sp.g_computation(df, y='wage', treat='trained',
                     covariates=['age', 'edu', 'exp'])

# Binary treatment — ATT
r = sp.g_computation(df, y='wage', treat='trained',
                     covariates=['age', 'edu'],
                     estimand='ATT')

# Continuous / multi-valued treatment — dose-response curve
r = sp.g_computation(df, y='bp', treat='dose',
                     covariates=['age', 'bmi'],
                     estimand='dose_response',
                     treat_values=[0, 10, 20, 30])
r.detail   # pd.DataFrame: dose / estimate / se / ci_lower / ci_upper / pvalue
```

**Learners**: defaults to OLS via statsmodels. Pass any sklearn-
compatible regressor via `ml_Q=` for flexible fits:

```python
from sklearn.ensemble import GradientBoostingRegressor
r = sp.g_computation(df, y='y', treat='d', covariates=[...],
                     ml_Q=GradientBoostingRegressor(n_estimators=200))
```

**Inference**: nonparametric bootstrap with NaN-based failure
tracking. Use `r.model_info['n_boot_failed']` to audit.

**Relationship to AIPW/TMLE**: `g_computation` is consistent when the
outcome model is correctly specified, but **not doubly robust**. Use
`sp.aipw` or `sp.tmle` when you want coverage guarantees from either
the outcome or the propensity model.

---

## `sp.front_door` — Pearl's front-door adjustment

Identifies `E[Y | do(D)]` when an unmeasured confounder blocks the
back-door criterion, but a mediator `M` fully transmits the effect of
D on Y:

```text
U ──┬──► D ──► M ──► Y
    │             ▲
    └─────────────┘     (U unobserved)
```

Front-door formula:

$$
E[Y | do(D=d)] = \sum_m P(M=m | D=d) \cdot \sum_{d'} P(D=d') \cdot E[Y | D=d', M=m].
$$

```python
# Binary mediator — closed-form sums
r = sp.front_door(df, y='y', treat='d', mediator='m',
                  covariates=['x'],
                  mediator_type='binary')

# Continuous mediator — Gaussian MC integration
r = sp.front_door(df, y='y', treat='d', mediator='m',
                  covariates=['x'],
                  mediator_type='continuous',
                  integrate_by='marginal',    # Pearl 1995 aggregate
                  n_mc=200)
```

**`integrate_by`** switches between two identification variants:

| Value | Interpretation |
| --- | --- |
| `'marginal'` (default) | Pearl (1995) aggregate — MC samples (X, M) jointly from the population-marginal distribution. |
| `'conditional'` | Fulcher et al. (2020) generalised front-door — M is drawn per unit from `P(M \| D=d, X_i)`. |

For no-covariate or binary-mediator problems the two coincide
(`model_info['integrate_by_effective']` makes this explicit).

**Continuous treatment**: currently rejected with a helpful error
pointing to `sp.g_computation(estimand='dose_response')` (see
ROADMAP §4).

---

## `sp.msm` — Marginal Structural Models (stabilised IPTW)

Robins (1998, 2000) for **time-varying** treatments whose
confounders are themselves affected by past treatment. Standard
regression adjustment biases the effect (blocks a causal path
*and* opens a collider); MSM decouples them via inverse-probability-
of-treatment weighting on a pooled outcome regression.

```python
# Long-format panel: one row per (unit, period)
r = sp.msm(panel, y='cd4_count', treat='hiv_therapy',
           id='patient_id', time='visit',
           time_varying=['cd4_lag', 'viral_load_lag'],
           baseline=['age', 'sex'],
           exposure='cumulative',    # 'cumulative' | 'current' | 'ever'
           trim=0.01,                # symmetric weight truncation
           trim_per_period=False)    # or True for Cole & Hernán §3
```

Stabilised weight for unit i at time t:

$$
sw_{i,t} = \prod_{s \le t}
   \frac{P(A_s = a_s | \bar{A}_{s-1}, V)}
        {P(A_s = a_s | \bar{A}_{s-1}, \bar{L}_s, V)}
$$

The pooled weighted regression of :math:`Y` on cumulative exposure
(and baseline V) recovers the **marginal** causal parameter with
cluster-robust CR1 standard errors clustered by unit.

**Weight diagnostics** live in `model_info`:

```python
r.model_info['sw_mean']       # should be ≈ 1 under correct spec
r.model_info['sw_max']        # extreme-weight watchdog
r.model_info['trim_per_period']
```

**Standalone helper**: `sp.stabilized_weights(...)` returns the
vector of row weights without fitting the outcome model — useful
for plugging into custom weighted estimators.

**Continuous / multi-valued treatment**: supported by Gaussian
density-ratio weights (`treat_type='continuous'`). `exposure='ever'`
requires binary A.

---

## `sp.mediate_interventional` — interventional effects

VanderWeele, Vansteelandt & Robins (2014) interventional direct and
indirect effects. Identifies the causal decomposition in the
presence of a **treatment-induced mediator-outcome confounder**,
where natural effects are not identified.

```python
r = sp.mediate_interventional(
    df, y='y', treat='d', mediator='m',
    covariates=['x_baseline'],
    tv_confounders=['L'],    # treatment-induced M-Y confounders
    n_mc=500, n_boot=500,
    pvalue_method='bootstrap_sign',   # or 'wald' (see below)
)

r.estimate   # IIE (interventional indirect effect)
r.detail     # IIE / IDE / Total breakdown
```

Decomposition:

* **IIE** (interventional indirect) :math:`= E[Y(1, G_{M|1})] - E[Y(1, G_{M|0})]`
* **IDE** (interventional direct) :math:`= E[Y(1, G_{M|0})] - E[Y(0, G_{M|0})]`
* **Total** :math:`= E[Y(1, G_{M|1})] - E[Y(0, G_{M|0})] = IIE + IDE`

where :math:`G_{M|d}` is a random draw from the marginal distribution
of :math:`M` under `D = d`, integrated over covariates.

**`pvalue_method`**:

* `'bootstrap_sign'` (default) — bootstrap CI-inversion p-value,
  matches `sp.mediate()` convention.
* `'wald'` — conventional :math:`2(1 - \Phi(|\hat\theta/\hat{se}|))`,
  matches the rest of the causal-inference surface (aipw, dml, etc.).

**Linearity assumption**: the current implementation hard-codes an
OLS outcome regression. This enables the MC integration to collapse
analytically via :math:`\beta_{TV} \cdot \mathrm{mean}(X_{TV})`,
giving an `O(n_mc + n)` runtime. Non-linear outcome models (GBM,
neural nets) would break this vectorisation and are not currently
supported.

---

## References

- Robins, J. (1986). A new approach to causal inference in mortality
  studies. *Mathematical Modelling*.
- Pearl, J. (1995). Causal diagrams for empirical research.
  *Biometrika*.
- Robins, Hernán & Brumback (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*.
- VanderWeele, Vansteelandt & Robins (2014). Effect decomposition in
  the presence of an exposure-induced mediator-outcome confounder.
  *Epidemiology*.
- Fulcher et al. (2020). Robust inference on population indirect
  causal effects: the generalized front-door criterion. *JRSS-B*.
- Cole & Hernán (2008). Constructing inverse probability weights for
  marginal structural models. *American Journal of Epidemiology*.
- Hernán & Robins (2020). *Causal Inference: What If*. Chapman &
  Hall/CRC — comprehensive textbook treatment of g-methods.
