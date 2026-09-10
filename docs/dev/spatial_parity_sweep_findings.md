# Spatial sweep — findings

Second family in the campaign described in
`cross_language_coverage_campaign.md`, after the RD sweep
(`rd_parity_sweep_findings.md`). Reference: R `spdep` 1.4.2 and
`spatialreg` 1.4.3, both already installed, which is why this family went
first among the eight.

All six entries below are closed in 1.27.0. The numbers are measurements
taken before the fix, on `tests/reference_parity/_fixtures/spdep_data.csv`.

**The recurring shape in this family: the comment above the line named the
right formula and the line below it computed something else.** That is
true of S1 and S2 independently, in the same function. It is a failure
mode unit tests are structurally bad at catching — the author knew the
formula well enough to write it down — and cross-language comparison
catches it immediately.

## S1 — `sp.lm_tests` T term: the comment is right, the code is not (⚠️ correctness)

`src/statspai/spatial/models/diagnostics.py:73-74`

```python
# T = tr(W' W + W W) — computed sparsely to avoid O(n²) dense allocation
T = float((M.T.multiply(M)).sum() + (M.multiply(M @ M.T)).sum())
```

The first term is `sum_ij W_ji W_ij` = tr(WW). The second is
`sum_ij W_ij (W W')_ij`, which is not tr(W'W) = `sum_ij W_ij^2`.

Correct: `T = (M.multiply(M)).sum() + (M.T.multiply(M)).sum()`.

On a row-standardised 10x10 rook lattice: 28.222 computed, 56.889 correct
— a factor of 2.02, so `LM_err` came out doubled (39.47 against spdep's
19.58).

## S2 — `sp.lm_tests` J term uses W y where Anselin uses W X beta (⚠️ correctness)

Same file, lines 79-81:

```python
# RS_lag: (e' W y / s2)^2 / [(WXb)'(M*WXb)/s2 + T]   with M = I - X(X'X)^-1 X'
MW_Xb = Wy_centered  # residual-ised Wy under OLS
```

The comment names `WXb`; the code substitutes `M(Wy)`. Since
`Wy = W X beta_hat + W e`, this mixes the residual's own spatial structure
into the denominator of every lag statistic.

## Consequence of S1 + S2 together

Against `spdep::lm.RStests` on the same weights and data:

| statistic | StatsPAI | spdep | after both fixes |
| --- | ---: | ---: | ---: |
| LM_err | 39.465383 | 19.578530 | 19.578530 |
| LM_lag | 20.150963 | 23.898571 | 23.898571 |
| Robust_LM_err | 20.489246 | 0.039707 | 0.039707 |
| Robust_LM_lag | 1.174826 | 4.359748 | 4.359748 |

The robust LM-error row is the one that matters most in practice: this is
the Anselin decision rule for choosing between a spatial lag and a spatial
error specification. StatsPAI reported 20.49 (p = 6e-6, "strong residual
error dependence") where the correct value is 0.0397 (p = 0.84, "none").
The battery was reversing the model-selection conclusion it exists to
inform.

## S3 — `sp.join_counts` BW is double the like-join convention (⚠️ correctness)

`src/statspai/spatial/esda/join_counts.py:65-68`

```python
bb = 0.5 * float(np.sum(data * ((y[rows] == 1) & (y[cols] == 1))))
ww = 0.5 * float(np.sum(data * ((y[rows] == 0) & (y[cols] == 0))))
bw = float(np.sum(data * (y[rows] != y[cols])))          # <- no 0.5
```

BB and WW halve the double sum; BW does not. The identity
`BB + WW + BW = S0/2` therefore fails: on the row-standardised 10x10 rook
lattice it gives 70.75 where S0/2 = 50. spdep's `joincount.multi` reports
BB 14.7917, WW 14.4583, BW 20.75; StatsPAI reported BW 41.5.

The identity is a self-contained proof and should be asserted as a test in
its own right, independently of spdep.

## S4 — `sp.geary` has no analytic variance

`geary(..., permutations=0)` returns `variance`, `z_score`, `p_norm` as
NaN, while `moran` returns all three. Geary's C has a closed-form
randomisation and normality variance (Cliff & Ord 1981), which
`spdep::geary.test` reports by default. Documented in the docstring, so
not silent, but it is a capability gap against the reference and the
asymmetry with `moran` is surprising.

## S6 — `sp.getis_ord_local(star=False)` standardises Gi with Gi*'s moments (⚠️ correctness)

`src/statspai/spatial/esda/getis_ord.py:170-184`

```python
mean_y = sum_y / n
var_y  = np.var(y, ddof=0)
num = S @ y - Wi * mean_y
denom_core = np.maximum((n * (Wi - Wi**2 / n)) / (n - 1), 0)
```

`mean_y` and `var_y` are whole-sample moments regardless of `star`, and
`denom_core` is the Gi* denominator. That is correct for Gi*, where
observation i is part of its own neighbourhood -- and `star=True` matches
`spdep::localG` on self-included weights to **1.7e-14**.

It is not correct for Gi. Ord and Getis (1995) standardise Gi with the
*exclude-self* moments

    xbar(i) = (sum_j x_j - x_i) / (n - 1)
    s(i)^2  = (sum_j x_j^2 - x_i^2) / (n - 1) - xbar(i)^2
    Gi      = (sum_j w_ij x_j - W_i xbar(i))
              / ( s(i) * sqrt( ((n-1) S_1i - W_i^2) / (n-2) ) )

which is what `spdep::localG` computes; reconstructing it independently
reproduces spdep to 4.3e-13. Against it, `star=False` is off by a relative
1.5 -- the numerator excludes i while the standardisation includes it, so
the two halves of the statistic disagree about what the neighbourhood is.

Same shape as the `sp.etregress` defect: one branch is right and the other
borrows its arithmetic.

## Verified-exact already (no fix needed)

| function | reference | agreement |
| --- | --- | ---: |
| `sp.moran` | `spdep::moran.test(randomisation=TRUE)` | I 1.9e-15, V 7.6e-15, z 1.8e-15 |
| `sp.geary` (value) | `spdep::geary.test` | 2.0e-15 |
| `sp.moran_local` | `spdep::localmoran` | 8.1e-15 |
| `sp.getis_ord_g` | `spdep::globalG.test` (binary W) | 5.4e-16 |
| `sp.join_counts` BB/WW | `spdep::joincount.multi` | exact |
| `sp.slx` | `spatialreg::lmSLX` | 1e-15 |
| `sp.sac` | `spatialreg::sacsarlm` | rho 3.9e-7, coefs ~1e-7 |

Note on weights: `sp.W` is binary until `w.transform = "R"`. The
comparison must set it, or the two sides are computing different
statistics. `getis_ord_g` matches spdep's BINARY-weight call, which is
what spdep itself recommends for that statistic.

## S5 — `sp.moran_residuals` uses the raw-variable null for OLS residuals

The statistic itself is exact: I = 0.3337365445780480 against
`spdep::lm.morantest`'s 0.3337365445780478, a relative gap of 5e-16.

The p-value is not. `moran_residuals` delegates to `moran(...)`, which
computes the null variance of Moran's I for an *observed variable*. Applied
to OLS residuals that null is the wrong one: the residuals are a projection
of y, and Cliff and Ord's E[I] and Var[I] for regression residuals depend
on X through the hat matrix. Measured: p = 3.64e-06 where `lm.morantest`
reports 1.56e-06, i.e. conservative by a factor of 2.3 here. The direction
is the harmless one on this fixture, but nothing guarantees that in
general, and the function's whole purpose is to be an "LM-err companion".

(The `TypeError` noticed first was my own: `EconometricResults.residuals`
is a method, not an attribute.)
