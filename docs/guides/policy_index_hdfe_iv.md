# Policy-intensity panels: HDFE-IV end to end

A recurring design in environmental and energy economics: a **policy-intensity
index** measured for every county in every month, a **high-dimensional
two-way fixed-effect** panel, an **interaction instrument** built from a
time-invariant exposure and a time-varying national shock, and inference that
has to survive clustering, spatial correlation, and seasonality.

The worked example below follows the shape of Zhang et al. (2026, *Science*
393:831-836, doi:10.1126/science.aee0747), who study how the intensity of
China's solar-expansion policy affects county-level bird diversity. Nothing
here is specific to solar panels or birds — the same skeleton fits any
"policy index × county-month panel × shift-share-style instrument" question.

Every number StatsPAI produces on this path is pinned against Stata
`ivreghdfe` / `ivreg2` / `ranktest` / `acreg` in
`tests/reference_parity/test_iv_hdfe_stata_parity.py`.

---

## 0. The design in one table

| Ingredient | This design | StatsPAI |
| --- | --- | --- |
| Outcome | Shannon diversity of the bird community, county × month | `sp.diversity_index` |
| Treatment | County policy-intensity index | — (constructed upstream) |
| Fixed effects | County + year-month | `sp.iv(absorb=[...])` |
| Instrument | Historical solar resource × inverse policy uncertainty | `sp.iv("y ~ (d ~ z) + x")` |
| Inference | Cluster by county | `cluster="county"` |
| Robustness | Spatial + serial correlation | `sp.conley(...)` |
| Weak-ID | Effective F, KP rk, AR set | `sp.iv_diag(...)` |
| Mechanisms | Same RHS, several LHS | loop + `sp.romano_wolf` |

---

## 1. Build the outcome

Citizen-science records arrive as one row per sighting. Turning them into a
panel outcome is a modelling decision, not data cleaning: Shannon entropy,
species richness and Pielou evenness answer different questions, and a policy
can leave Shannon flat while pushing richness down and evenness up.

```python
import statspai as sp

diversity = sp.diversity_index(
    records,                       # one row per sighting
    species="species",
    by=["county", "ym"],
    index=["shannon", "richness", "pielou"],
    min_records=5,                 # thin county-months return NaN, explicitly
)
panel = panel.merge(diversity, on=["county", "ym"], how="left")
```

`min_records` matters. Diversity indices are biased downward in small samples,
so a county-month with three sightings is not a low-diversity county-month —
it is an unmeasured one. Setting the floor explicitly keeps that decision in
the code rather than in a footnote.

The same call also returns `n_records`, which belongs on the right-hand side:
with citizen-science data, observation effort enters the outcome mechanically.

---

## 2. The baseline: two-way FE, clustered

```python
res = sp.iv(
    "shannon ~ (policy ~ z) + temp + wind + log_pop + log_birding_hours",
    data=panel,
    absorb=["county", "ym"],       # HDFE, partialled out (Rust backend)
    cluster="county",
)
print(res.summary())
```

`absorb=` residualises the outcome, the controls, the endogenous regressor
*and* the instrument before estimating, so this is `ivreghdfe`, not a
regression with thousands of dummies. It works for every estimator —
`method="liml"`, `"fuller"`, `"gmm"`, `"jive"` — accepts multiway
clustering, `cluster=["county", "ym"]`, and takes interacted fixed effects
in the fixest spelling, `absorb=["county", "prov^year"]` (province-by-year
FE is the standard way to soak up regional policy waves).

### The degrees-of-freedom rule you are relying on

Clustering on the same dimension you absorb (county FE, county clusters) is
the single most common panel specification, and it has a subtlety: a fixed
effect **nested within a clustering dimension** costs no residual degrees of
freedom, because the cluster sums already annihilate it. `reghdfe`
(`dofadjustments(clusters)`) and `fixest` (`fixef.K="nested"`) both drop it.
StatsPAI does too, and reports what it did:

```python
res.model_info["fe_nested_in_cluster"]   # ['county']
res.model_info["fe_dof_charged"]         # 23  == ivreg2's e(sdofminus)
```

Charging those degrees of freedom anyway inflates standard errors by roughly
`sqrt((N-k)/(N-k-G_county))` — several percent in a typical county panel, in
the direction that makes results look weaker than the reference
implementation says they are.

---

## 3. Is the instrument strong? Read three numbers, not one

```python
diag = sp.iv_diag(
    panel,
    y="shannon", endog="policy", instruments=["z"],
    exog=["temp", "wind", "log_pop", "log_birding_hours"],
    absorb=["county", "ym"],
    cluster="county",
)
print(diag.summary())
```

The bundle reports, in the vcov you actually estimated:

* **Olea-Pflueger effective F** — the pre-test for the concentration
  parameter. With one instrument it coincides with the KP rk Wald F.
* **Kleibergen-Paap rk LM** — the *under*identification test.
* **Kleibergen-Paap rk Wald F** — the *weak*identification test.
* **Anderson-Rubin set** — size-correct whatever the first stage looks like.

All four follow the estimator's own variance: cluster the fit and the rank
tests cluster too. This matters more than it sounds. A heteroskedasticity-only
first-stage F sitting next to cluster-robust coefficients is measuring
instrument strength under an assumption the rest of the table has already
abandoned, and it is biased toward looking strong.

The AR confidence set is the one to quote when the effective F is anywhere
near the Stock-Yogo region:

```python
ar = sp.anderson_rubin_test(
    data=panel, y="shannon", endog="policy", instruments=["z"],
    exog=["temp", "wind"], absorb=["county", "ym"], cluster="county",
)
ar["ar_ci"], ar["ar_ci_disjoint"]
```

`ar_ci_disjoint` flags the case where the AR set is a union of intervals
rather than one — a real possibility under weak identification, and something
a convex hull would quietly hide.

### Over-identified specifications

Add a second instrument and the over-identification test appears
automatically — **Sargan** under i.i.d. errors, **Hansen J** as soon as the
vcov is robust or clustered, exactly as `ivreg2` switches. Reading a Sargan
statistic off a clustered regression is reading a test whose null distribution
assumed away the clustering.

---

## 4. Spatial and serial correlation

Counties are not independent draws, and neither are consecutive months.
`sp.conley` takes the fitted IV result directly:

```python
spatial = sp.conley(res, panel, lat="lat", lon="lon", dist_cutoff=200)

spacetime = sp.conley(
    res, panel, lat="lat", lon="lon", dist_cutoff=200,
    time="ym", lag_cutoff=12, unit="county",   # 12-month serial window
)
```

Pass `distance="planar"` to reproduce Stata `acreg`'s convention exactly;
the default `"haversine"` is great-circle and symmetric.

A practical note on ordering: run Conley on the **absorbed** fit. The kernel
is applied to the 2SLS scores of the residualised design, which is what the
`acreg ... pfe1() pfe2()` route computes.

---

## 5. Heterogeneity and mechanisms

Subsample splits are just re-fits, and the honest way to compare them is
side by side with their intervals:

```python
rows = []
for name, mask in {
    "non-poor": panel.poor == 0,
    "poor": panel.poor == 1,
}.items():
    r = sp.iv(FORMULA, data=panel[mask], absorb=["county", "ym"], cluster="county")
    rows.append({"group": name, "beta": r.params["policy"], "se": r.std_errors["policy"]})
```

Mechanism outcomes share the right-hand side, so they are a family of
hypotheses, not one: adjust for it.

```python
sp.romano_wolf(
    data=panel, y=["ndvi", "lai", "nightlight"], x="policy",
    controls=["temp", "wind"], cluster="county",
)
```

Two mechanism outcomes moving in *opposite* directions — vegetation index
down, leaf-area index up — is a substantive finding, not a contradiction, and
it survives multiplicity adjustment or it does not. Report which.

---

## 6. Testing the exclusion restriction where you can

An interaction instrument inherits the exclusion restriction of both legs,
and exclusion is not testable in the estimation sample. It *is* testable in a
subsample where the instrument has no first stage — a desert county for a
solar-resource instrument, an industry with no national shock for a Bartik
one. If the instrument still moves the outcome there, it is moving it through
something other than the treatment.

```python
zfs = sp.zero_first_stage(
    panel, y="shannon", endog="policy", instrument="z",
    zfs="is_desert",                 # the inert subsample
    exog=["temp", "log_effort"], absorb=["county", "ym"], cluster="county",
)
print(zfs.summary())
```

The result reports three things, in this order:

1. **the premise** — the first stage really is ~0 in that subsample, with an
   interval, because "insignificant" is not "zero";
2. **the test** — the reduced form there, which *is* the direct effect
   estimate under the premise;
3. **the consequence** — the implied bias `gamma / pi` in the main-sample IV
   estimate, and van Kippersluis–Rietveld's corrected estimate that nets it
   out (assuming the direct effect is common to both subsamples).

A failure to reject is not a clean bill of health. Quote the confidence
interval on the direct effect: it is the set of violations the data cannot
rule out.

---

## 7. What still needs care

* **The index is not the policy.** A policy-intensity index built from
  document counts weighted by administrative rank is a measurement model with
  its own assumptions. Validate it against physical deployment (installed
  capacity, land footprint) and report the lag structure of that correlation.
* **Effort confounds citizen-science outcomes.** Control observation effort
  and show the result survives dropping thin cells (`min_records`).
* **Exclusion, continued.** Where no zero-first-stage subsample exists,
  `sp.iv(method="plausibly_exog_ltz", ...)` puts a prior on the direct effect
  and reports how large it would have to be to overturn the result — cheaper
  than arguing about it in prose.
* **Interpretation.** With heterogeneous effects, 2SLS with covariates is a
  weighted average that need not be the ATE; `sp.iv_diag` prints the
  TSLS-vs-LATE caveat when the endogenous regressor is binary.

---

## Reference-implementation parity

| StatsPAI | Stata | Agreement |
| --- | --- | --- |
| `sp.iv(absorb=, cluster=)` | `ivreghdfe ..., absorb() cluster()` | machine precision |
| `sp.iv(absorb=, cluster=[a, b])` | `ivreghdfe ..., cluster(a b)` | machine precision |
| `sp.iv(absorb=, method="liml"/"fuller")` | `ivreghdfe ..., liml/fuller(1)` | coefficients exact; SEs differ by `O(kappa-1)` (documented convention gap) |
| `sp.iv(absorb=, method="gmm", gmm_vcov="efficient")` | `ivreghdfe ..., gmm2s` | machine precision |
| `KP rk LM` / `KP rk Wald F` | `ranktest` / `e(idstat)`, `e(widstat)` | machine precision |
| `Sargan` / `Hansen J` | `e(sargan)` / `e(j)` | machine precision |
| `sp.effective_f_test(vcov="classic")` | `e(cdf)` (Cragg-Donald) | machine precision |
| `sp.conley(..., distance="planar")` | `acreg ..., spatial pfe1() pfe2()` | machine precision |

---

## References

- Anderson, T. W. and Rubin, H. (1949). "Estimation of the Parameters of a
  Single Equation in a Complete System of Stochastic Equations." *Annals of
  Mathematical Statistics*. [@anderson1949estimation]
- Baum, C. F., Schaffer, M. E. and Stillman, S. (2007). "Enhanced routines for
  instrumental variables/generalized method of moments estimation and
  testing." *The Stata Journal*, 7(4), 465-506. [@baum2007enhanced]
- Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2011). "Robust Inference
  With Multiway Clustering." *JBES*, 29(2), 238-249. [@cameron2011robust]
- Colella, F., Lalive, R., Sakalli, S. O. and Thoenig, M. (2023). "acreg:
  Arbitrary correlation regression." *The Stata Journal*. [@colella2023acreg]
- Conley, T. G. (1999). "GMM estimation with cross sectional dependence."
  *Journal of Econometrics*. [@conley1999estimation]
- Kleibergen, F. and Paap, R. (2006). "Generalized reduced rank tests using
  the singular value decomposition." *Journal of Econometrics*.
  [@kleibergen2006generalized]
- van Kippersluis, H. and Rietveld, C. A. (2018). "Pleiotropy-robust
  Mendelian randomization." *International Journal of Epidemiology*, 47(4),
  1279-1288. [@vankippersluis2018pleiotropy]
- Shannon, C. E. (1948). *Bell System Technical Journal*, 27(3), 379-423.
  [@shannon1948mathematical]
- Pielou, E. C. (1966). *Journal of Theoretical Biology*, 13, 131-144.
  [@pielou1966measurement]
- Hill, M. O. (1973). *Ecology*, 54(2), 427-432. [@hill1973diversity]
