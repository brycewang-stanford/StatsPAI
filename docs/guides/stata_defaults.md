# Reproducing Stata defaults

Most "StatsPAI gives a different number" reports on replication packages
come from a Stata default rather than an estimator. This page lists the
defaults that decide `N`, the sample, or a threshold, and the StatsPAI
argument that corresponds to each. Every row is checked against Stata 18
by a test in `tests/reference_parity/`.

## Commands and options

| Stata | StatsPAI | Notes |
| --- | --- | --- |
| `ppmlhdfe y x, absorb(id ind#year) vce(cluster city#year)` | `sp.ppmlhdfe("y ~ x", data=df, absorb="id + ind#year", cluster="city#year")` | `^` (fixest) and `#` (Stata) both build interacted groups. |
| `ppmlhdfe` singleton and separation drops | default (`drop_singletons=True`, `separation=True`) | `model_info['n_singletons']` + `['n_separated']` equal Stata's `e(num_singletons)`. `drop_singletons=False` matches R `fixest::fepois`. |
| `ppmlhdfe` "(omitted)" regressors | automatic | Regressors absorbed by the fixed effects are listed in `model_info['omitted']`. |
| Hand-built stack + `ppmlhdfe y tp x, absorb(id#cohort year#cohort ...)` | `sp.stacked_did(df, y=..., group=..., time=..., first_treat=..., family="poisson", spec="pooled", absorb="...", control_group="notyettreated_rows")` | Later-treated units serve as controls only before their own treatment. `spec="event_study"` gives the event-time coefficients. |
| `reghdfe ..., vce(cluster c)`: `e(r2)`, `e(r2_a)`, `e(r2_within)`, `e(df_a)`, `e(df_a_nested)` | `sp.hdfe_ols(...).r2`, `.r2_a`, `.r2_within`, `.df_a`, `.df_a_nested` | `r2_a` charges effects nested in the cluster, as `reghdfe` does. |
| `winsor2 v, cuts(1 99)` | `sp.winsor(df, ["v"], cuts=(1, 99))` | Stata `_pctile` percentiles by default; `method="r"` for R's `quantile(type = 2)`, `method="linear"` for numpy's. |
| `winsor2 v if year>=2007, by(g)` | `sp.winsor(df, ["v"], subset="year >= 2007", by="g")` | Outside `subset`, new columns are missing and `replace=True` leaves values unchanged. |
| `logit y x i.g` ("predicts failure perfectly") | `sp.logit("y ~ x + C(g)", data=df)` | Default `perfect_prediction="drop"`; `"keep"` keeps the rows. Same for `probit` / `cloglog`. |
| `psmatch2 d x, outcome(y) neighbor(1) ties ate common` | `sp.psmatch2(df, treat="d", covariates=["x"], outcome="y", common_support="minmax", ties=True, ate=True)` | PSM-DID on `_weight != .` uses `m.matched_data["_weight"].notna()`. ATU / ATE are in `m.result.model_info`. |
| `xtreg y d x i.ind#i.year, fe vce(cluster c)` then `psacalc` with `rmax(1.3*e(r2_a))` | `sp.oster_bounds(df, y="y", treat="d", controls=["x"], absorb="id", absorb_controls="ind#year", cluster="c", r_max="1.3*r2_a")` | Exact Oster solution. `moments=` gives the exact solution from summary statistics. |
| `csdid2 y, ivar(id) time(t) gvar(g)` on an unbalanced panel | `sp.callaway_santanna(..., allow_unbalanced_panel=True)` | The default (`False`) keeps only units observed in both periods of a comparison and gives a different point estimate. |
| `csdid` 2.0.0 or R `did::att_gt` on an unbalanced panel, at their defaults (`bal(full)`, `allow_unbalanced_panel = FALSE`) | `sp.callaway_santanna(..., balance="full")` | Both drop every unit not observed in all periods before estimating. StatsPAI's default is `balance="pair"`, the rule of `csdid` 1.8x (`bal(pair)` in 2.0.0). |
| `csdid y, time(t) gvar(g) cluster(c) agg(group)` without `ivar()` (rows as repeated cross-sections) | `sp.callaway_santanna(..., panel=False, clustervars="c")` then `sp.aggte(r, type="group", agg_weights="csdid", share_variance=False)` | Any row-level cluster, e.g. `city#year`. Equal to `csdid` v1, including its estimated cell-count weights. |
| `csdid2` with no never-treated units | `control_group="notyettreated"` | `csdid2` switches to not-yet-treated controls without saying so; StatsPAI asks for it explicitly. |

## Textbook commands (Wooldridge, Mixtape)

Checked against Stata 18 MP on the books' original `.dta` files in
`tests/test_textbook_parity_nov28.py` (runs when `STATSPAI_WOOLDRIDGE_DIR` /
`STATSPAI_MIXTAPE_DIR` point at the files).

| Stata | StatsPAI | Notes |
| --- | --- | --- |
| `newey y x, lag(2)` | `sp.regress("y ~ x", df, robust="newey", hac_lags=2)` | `robust="hac"` alone omits the `N/(N-K)` factor (R `NeweyWest(adjust = FALSE)`, statsmodels); `"newey"` is `"hac"` plus `hac_small=True`. |
| `estat ovtest` | `sp.estat(r, "reset", powers=4)` | The default `powers=3` is Wooldridge's and R `lmtest::resettest`'s. Stata builds the powers in single precision, so the F agrees to about 1e-6. |
| `estat hettest` | `sp.estat(r, "hettest", variables="fitted", version="normal")` | The default is Koenker's N R² on all regressors, Stata's `estat hettest, rhs iid`. |
| `ivregress 2sls y (x = z) w` | `sp.ivreg("y ~ (x ~ z) + w", df, small=False)` (or `sp.iv(..., small=False)`) | The default is `ivregress ..., small` (t and F, `N - K`). |
| `xtreg ..., fe` / `xtreg ..., re` / `hausman fe re` | `fe.hausman_test(efficient=re)` or `sp.hausman(fe, re)` | Without `efficient=`, `hausman_test()` refits RE on the FE regressors; with time-invariant regressors in your RE model that is a different model. |
| `xtreg ..., re theta`: `e(sigma_u)`, `e(sigma_e)`, `e(rho)`, `e(theta)` | `re.model_info["sigma_u"]`, `["sigma_e"]`, `["rho"]`, `["theta"]` | `theta` is a scalar on a balanced panel, else its min / median / max. |
| `regress D.y D.x` after `xtset` | `sp.panel(df, "y ~ x", entity=..., time=..., method="fd", constant=True)` | The default is `regress D.y D.x, nocons`. The intercept is `const`. |
| `heckman ..., twostep`: `e(N)`, `e(N_selected)` | `res.n_obs`, `res.model_info["n_selected"]` | Both methods report the full sample as `n_obs` (two-step reported the selected rows through 1.46.7). |
| `test x1 x2` with a term omitted as collinear ("constraint dropped") | `sp.test(r, "x1 x2", omitted="skip")` | The default refuses the test. `out["omitted_dropped"]` lists the dropped restrictions. `sp.stata("test ...")` translates with `omitted="skip"`. |
| `regress y x [aweight=w]` with some `w == 0` | `sp.regress("y ~ x", df, weights="w")` | Zero-weight rows leave the sample with a note (`data_info["n_zero_weight_dropped"]`), as in Stata and R `fixest`. Negative weights are refused. `sp.rdrobust` keeps them, as R `rdrobust` does. |
| `jive y (x = z1-z8) w1-w6, robust` with collinear dummies | `sp.jive(df, y=..., x_endog=[...], x_exog=[...], z=[...], robust="hc1")` | Collinear exogenous regressors and instruments are omitted in Stata's order and listed in `model_info["omitted"]`. Same for `sp.liml`. |
| `synth ..., mspeperiod(1985(1)1993)` | `sp.synth(..., mspe_period=list(range(1985, 1994)))` | The window the outcome MSPE (and the regression-based `V`) is computed over. The default is the pre-treatment periods. The in-space placebos keep the treated unit in each donor pool, as the ADH loop does. |
| A `%td` date as a regressor | Convert first: `df["d"] = (df["d"] - pd.Timestamp("1960-01-01")).dt.days` | A `datetime64` column in a formula is refused: patsy would make one dummy per date. Read with `pd.read_stata(..., convert_dates=False)` to keep Stata's day count. `C(date)` still asks for date dummies. |
| `regress` keeps the first of a collinear set | `sp.regress` omits the last | The coefficients of the kept columns are the same; the table lists a different dummy as omitted. |
| `synth` builds the synthetic path from weights rounded to three decimals, `e(RMSPE)` from unrounded ones | Unrounded throughout | Round `model_info["weights"]` yourself to print Stata's synthetic series. |
| `regtable` of models with different N (lags) | `sp.regtable(..., consistency_check=False)` | Turns off the sample-size note. |

## Known remaining differences

- **Propensity scores from `logit`.** StatsPAI and Stata agree to about
  1e-7, which is Stata's convergence tolerance. Nearest-neighbour matching
  on 10^5 scores can switch a few hundred matches at that scale. Given the
  same score, `sp.psmatch2` reproduces `psmatch2` exactly.
- **`xtreg, fe`'s `e(r2_a)` with thousands of dummies.** StatsPAI counts the
  control dummies at their exact rank given the panel effect. On one
  replication, Stata's figure implied three more regressors than that rank.
  Pass Stata's number as `r_max=` when a table must be reproduced digit for
  digit.
- **`csdid2`'s standard errors on repeated cross-sections.** `csdid2`
  (without `ivar()`) gives the same point estimates as `csdid` and StatsPAI,
  but smaller SEs. Its influence function for control rows is scaled by the
  share of the whole 2×2 subsample instead of the control cell's share.
  StatsPAI follows `csdid` v1 and R `DRDID`, which agree with each other.
  On one replication this moved a z statistic from 3.171 (`csdid2`) to
  3.113 (`csdid`, StatsPAI).
