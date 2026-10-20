# Clean-room specification: `diagnostics/_weakiv_lm.py`

Audit row M5 of `docs/dev/2026-10-20-license-provenance-audit.md`. The
previous module described itself as following the internals of Stata
`weakiv` (Finlay, Magnusson, Schaffer; no licence stated). It was replaced
on 2026-10-20 by a rewrite from the inputs listed below. The author of the
rewrite did not read the old function bodies, their git history, or any
`weakiv` / `ivreg2` / `ranktest` / `avar` source (`.ado`, `.mata`, `.mo`).

## Sources

* Papers (bib keys in `paper.bib`): `anderson1949estimation` (AR test),
  `kleibergen2002pivotal` and `kleibergen2005testing` (K statistic, the
  orthogonalised first-stage estimator `D`, the J split of AR),
  `moreira2003conditional` (CLR statistic and its conditional law),
  `kleibergen2006generalized` (the SVD-based rank statistic).
* Stata `weakiv` 2.4.07 help file (`weakiv.sthlp`, installed from SSC into a
  private ado directory): LM versus MD forms, the K-J p-value formula, the
  CLR for one regressor by a fast closed-form algorithm and by simulation
  for several, the subset AR test (i.i.d. only), projection sets.
* Black-box runs of Stata 18 MP with `weakiv` 2.4.07 / `ivreg2` / `ranktest`
  / `avar` on `tests/reference_parity/_fixtures/weakiv_fx.csv`
  (`_generate_weakiv_Stata.do`, values committed in
  `tests/reference_parity/test_weakiv_stata.py`; `e(rk_df)` read from the
  printed `ereturn list`).
* Outputs of the previous StatsPAI module, called as a black box on a
  battery of inputs (never its source), to check the API contract and the
  numbers.

## Inputs and estimands

`y` (n), `X` (n x p) endogenous, `Z` (n x k) excluded instruments, all
already residualised on the included exogenous regressors. Null
`H0: beta = b0`. Requires `k >= p` (else `IdentificationFailure`).

Reduced form `Y = [y, X] = Z B + U`, `B = (Z'Z)^{-1} Z'Y`.
`a = (1, -b0')'`, `e0 = Y a`, moment `r = B a = (Z'Z)^{-1} Z'e0`,
`Pi = B[:, 1:]`. With `V` the estimated variance of `vec(B)` (column
stacked), `psi = Var(r)`, `C_j = Cov(Pi_j, r)`:

* `D_j = Pi_j - C_j psi^{-1} r` (Kleibergen's estimator, uncorrelated with `r`);
* `S_D = Var(vec Pi) - C psi^{-1} C'`;
* `AR = r' psi^{-1} r` ~ chi2(k);
* `K = r' psi^{-1} D (D' psi^{-1} D)^{-1} D' psi^{-1} r` ~ chi2(p);
* `J = AR - K` ~ chi2(k - p) (absent when `k = p`);
* `CLR = (AR - rk + sqrt((AR + rk)^2 - 4 J rk)) / 2`;
* `rk`: Kleibergen-Paap statistic for `H0: rank = p - 1`, df `k - p + 1`.

### Variance of `vec(B)`

| `vcov` | `V` | residuals in the scores |
| --- | --- | --- |
| `'iid'` | `Sigma (x) (Z'Z)^{-1}`, `Sigma = U'U / N` | unrestricted reduced-form `U` |
| `'robust'` | `(I (x) (Z'Z)^{-1}) M (I (x) (Z'Z)^{-1})`, `M = sum_i h_i h_i'` | `h_i = [e0_i, x_i'] (x) z_i` (uncentred) |
| `'cluster'` | same with `h_g = sum_{i in g} h_i` | same |

No small-sample factor in any case (`weakiv` without `small`).

Black-box evidence (one regressor, null 0.5, `weakiv_fx.csv`): with the
robust and cluster variances, scores built from the restricted residual
`e0` and the raw regressors reproduce Stata's AR, K, J and rk to 3e-15;
unrestricted residuals miss AR by 1.5e-2 (robust) and 2.8e-2 (cluster), and
centred regressor scores miss rk by 0.1 to 1.2. Under i.i.d. the
unrestricted reduced-form `Sigma` reproduces rk exactly, and AR, K, J match
after the LM rescaling below.

### LM form under i.i.d.

The help file says the default for linear models is the LM version. Under
i.i.d. errors AR, K and J above (the MD / Wald version) are multiplied by
`e0'M_Z e0 / e0'e0`, so `AR = N e0'P_Z e0 / e0'e0`. `rk` and `CLR` keep the
MD values: Stata's `clr_stat` 1.6600334 equals the formula with the MD AR
and J (with the LM values it would be 1.6514).

### The rank statistic

`rk` is the Kleibergen-Paap statistic (`kleibergen2006generalized`) applied to `Theta = L D` with
`L` the lower-triangular Cholesky factor of `psi^{-1}` (`L L' = psi^{-1}`)
and no column normalisation; `Var(vec Theta) = (I_p (x) L) S_D (I_p (x) L)'`.
For `Theta = U S V'` and `U_q`, `V_q` the singular vectors beyond the first
`q = p - 1`, `rk = lam' Omega^{-1} lam`, `lam = vec(U_q' Theta V_q)`,
`Omega = (V_q' (x) U_q') Var(vec Theta) (V_q (x) U_q)`. (The paper's
null-space bases differ from `U_q`, `V_q` by orthogonal rotations, which
leave the quadratic form unchanged.)

With `p = 1` the statistic is `D' S_D^{-1} D` whatever the left factor. With
`p >= 2` it depends on the factor through `L'L`: black-box comparison on the
six two-regressor cases (two nulls x three variances) gave relative errors
of 2e-4 to 3e-3 for the symmetric square root of `psi^{-1}`, 3e-6 to 2e-2
for `(Z'Z)^{1/2}`, 1e-3 to 2e-2 for the identity, and below 7e-16 in all six
for the lower Cholesky factor. The choice is therefore pinned by
observation; it makes `rk` depend on the order of the instruments when
`p >= 2`, as `weakiv`'s does.

## CLR p-value

Given `rk`, under H0 `K ~ chi2(p)` and `J ~ chi2(k - p)` are independent
(`moreira2003conditional` for `p = 1`; the same argument for `p >= 2`). CLR is
increasing in `K` for fixed `J` and `rk`, and solving `CLR(K, J) = c` gives
`K = c - J c / (rk + c)`. Hence, with `s = c / (rk + c)`,

`P(CLR > c | rk) = 1 - P(K + s J <= c) = 1 - int_0^c f_p(x) F_{k-p}((c - x)/s) dx`.

The integrand is `x^{p/2-1} (c - x)^{(k-p)/2} g(x)` with `g` smooth
(`F_nu(y) / y^{nu/2}` is an entire function of `y`), so QUADPACK's
algebraic-weight rule (`scipy.integrate.quad(weight="alg")`) converges to
about 1e-16; tested against 40-digit `mpmath` integration over `J` (the
other order). Limits: `c <= 0` -> 1, `c = inf` -> 0, `k = p` or
`rk = inf` -> `P(chi2(p) > c)`, `rk = 0` -> `P(chi2(k) > c)`, NaN -> NaN.

Since `K <= CLR <= AR` pointwise (CLR is decreasing in `rk`), the p-value
lies between `P(chi2(p) > CLR)` and `P(chi2(k) > CLR)`. `at(b0,
clr_alpha=a)` returns the deciding bound when it already settles the test
at level `a`, and the exact value otherwise; this only speeds up grid
inversion. `clr_p=False` returns NaN.

`k = p`: no J (the `j` entries are `None`), `K = AR`, the CLR test is the AR
test and `clr_p = ar_p`. Under i.i.d. the reported `clr` statistic is the MD
AR while `ar` is the LM AR (same convention as the overidentified case).

Differences from Stata, by design: for one regressor `weakiv` evaluates the
same conditional law by a 20-node Simpson rule (a fixed-node closed-form
algorithm per the help file), so its p-values differ from the exact ones by ~1e-7; for several
regressors it simulates the CLR distribution drawing AR and K as
independent chi-squares although `AR = K + J`, a different law (0.80 where
the exact value is 0.75 on the fixture). Both are pinned in
`test_weakiv_stata.py`.

## Subset and projection AR (i.i.d. only)

For `H0: beta_1 = b1` with `beta_2` (on `X2`, p2 columns) free:
`Y0 = [y - x1 b1, X2]`, `mu = lambda - 1` the smallest root of
`|Y0'P_Z Y0 - mu Y0'M_Z Y0| = 0` (LIML), so `mu = min_{b2} e'P_Z e / e'M_Z e`.

* subset AR (`weakiv, subset()`; the help file gives its source): `(N - L) mu` against chi2(k - p2), `L` = number
  of excluded plus included instruments. Black box: `(N - L)` matches
  Stata's `e(ar_chi2)` to 5e-12 at b1 in {0.2, 0.5, 1.5} and for the other
  coefficient; `N`, `N - k`, `N - 2` miss by 0.5% to 1.3%.
* projection AR: `min_{b2}` of the joint LM AR `N e'P_Z e / e'e`, which is
  `N mu / (1 + mu)`, against chi2(k) (closed form; the help file's
  projection sets are grid approximations of the same set).

`mu` is computed directly as the smallest generalised eigenvalue (not as
`lambda - 1`), which avoids cancellation when `mu` is small.

## Confidence sets for a scalar coefficient

`{b : p(b) > alpha}` is found on a grid: `n_dense` (default 2001) equally
spaced points on `center +/- 50 scale` plus, on each side, `n_tail - 1`
(default 199) points spaced geometrically out to `center +/- 1e6 scale`
(`center`, `scale` = 2SLS estimate and its standard error, supplied by the
caller). Maximal runs of accepted points give intervals; each finite
endpoint is the root of `p(b) - alpha` between the neighbouring grid points
(Brent, tolerance ~1e-15 relative); a run touching the grid's end is
unbounded (`+/- inf`). No accepted point gives an empty list. NaN p-values
count as rejections. These are the same grid extents as the previous
module, read from its `_scan_grid` output.

## API kept

`_rk_kp(D, G, W, q)`, `clr_pvalue(clr, rk, k, p)`, `_cluster_sums(A,
codes)`, `WeakIVLM(y, X, Z, *, vcov, cluster_codes).at(b0, clr_alpha=None,
clr_p=True)` (dict keys in the order `ar, ar_df, ar_p, k, k_df, k_p, rk,
rk_df, rk_p, clr, j, j_df, j_p, clr_p`), `weakiv_tests(...)`,
`liml_lambda(Y0, Z)`, `SubsetAR(y, x1, X2, Z, L)` with attributes `y, x1,
X2, Z, n, k, p2, L` and methods `lam`, `subset`, `projection` (each returning
`{'stat', 'df', 'p'}`), `_scan_grid`, `_intervals_from_grid`,
`invert_pvalues`, `invert_pvalue`.

Behaviour changes at the edges (no caller reaches them; `_weakiv_multi`
validates first): `k < p` now raises `IdentificationFailure` (was a silent
result with NaN rk); a wrong-length or non-finite `b0` raises
`MethodIncompatibility` (was a NumPy `ValueError` / `LinAlgError`);
`cluster` with fewer clusters than instruments raises `DataInsufficient`;
`clr_pvalue` returns the correct limits for `clr = inf` and `rk = inf` (was
NaN); 1-d inputs are accepted by `_cluster_sums` and `liml_lambda`.
