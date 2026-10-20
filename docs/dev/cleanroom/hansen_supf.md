# Clean-room specification: Hansen (1997) sup-F p-values

Audit row M6 of `docs/dev/2026-10-20-license-provenance-audit.md`: the
25 x 3 x 40 coefficient table `HANSEN_SUP_BETA` was "transcribed from
Hansen's own `pv_sup.R`" (distributed from his web page with no licence
statement; the same table ships in R `strucchange`, GPL), and
`hansen_supf_pvalue` "transcribes `pv_sup`".

## Is the table published data?

Only in part. The article (JBES 15(1), 60-67, read from the author's PDF)
prints the coefficients for pi_0 = .01, .05, .15, .25, .35 and selected m,
to two decimals (Table 2), and says (p. 65): "I still have over 9,000
coefficients to report, which is too many to print in this article. The
complete estimates are available in a GAUSS program." The full-precision
table used by the code is therefore not published in the paper, so it was
recomputed.

## Recomputation (`scripts/generate_hansen_supf_table.py`)

Taken from the paper only:

- Statistic (eqs. 2, 5): `SupF(pi_0) = sup B(tau)'B(tau) / (tau(1 - tau))`
  over `pi_0 <= tau <= 1 - pi_0`, `B` an m-vector Brownian bridge.
- Simulation (Section 2.3): Brownian motion on 1,000 evenly spaced points;
  pi_0 = .01, .03, ..., .49; m = 1, ..., 40. We use 200,000 replications
  (four times the paper's 50,000; a first 50,000-replication table was
  slightly liberal for m = 2), seed 19970101; one draw of 40 independent
  bridges per replication serves every m (partial sums over the
  components).
- Approximating family (eq. 8, `v = 1` for SupF):
  `p(x | theta) = 1 - chi2(theta_0 + theta_1 x; eta)`.
- Fit (eqs. 11-12): minimise `[sum_q |p(Q(q) | theta) - q|^8 w(q)]^(1/8)`
  over `q = .001, ..., .999`, `w = 1` on [0, .1], `((.8 - q)/.7)^2` on
  [.1, .8], 0 above; `Q` the empirical upper-quantile function. Nelder-Mead,
  started at `(0, 1, m)` (the exact chi2(m) at pi_0 = .5) and warm-started
  across pi_0.

Checks: against the printed Table 2 the p-values lie within the range
spanned by the printed coefficients' rounding, widened by 2 x 0.0058 (the
maximum error Hansen's Table 1 reports); against Hansen's full table (R
`strucchange` and Stata `estat sbsingle` outputs in
`tests/reference_parity/_fixtures/r2_ts_*.json`) the largest |dp| is 0.0006
for p in [.001, .01), 0.0020 in [.01, .05), 0.0021 in [.05, .10) and 0.0076
above, i.e. within twice Hansen's own approximation error (0.0025 for
p <= .10, 0.0058 overall). Exact agreement is impossible by construction
(two independent Monte Carlo approximations). Against an independent
100,000-replication simulation (seeds 7 and 8) at pi_0 = .15, the p-value
the table assigns to the true 10% / 5% / 1% points is, for m = 1, 2, 3, 5:
.0996/.0501/.0095, .1009/.0501/.0091, .0996/.0491/.0093, .1004/.0496/.0098
(Hansen's table: .0999/.0504/.0096, .1008/.0502/.0092, .0978/.0479/.0088,
.1001/.0501/.0103), so the recomputed table is as accurate as the original.

## p-value function (`hansen_supf_pvalue`)

Written from eq. (8) and black-box behaviour of the previous function (which
matches R `strucchange` and Stata to 1e-10): at a tabulated pi_0 the row's
eq. (8) value; between rows, linear interpolation of the p-values in pi_0;
between .49 and .5, linear interpolation towards the exact `chi2(k)` value
at .5; below .01 the .01 row. `lam >= 1` is Andrews' lambda,
`pi_0 = 1 / (1 + sqrt(lambda))`. Run on the previous coefficients, the new
function reproduces the previous one to 7.7e-15 on 20,000 random calls.
Two defects of the previous function are now refused with
`MethodIncompatibility`: `k = 0` (was a `KeyError`), non-integer `k` (was
truncated), and a trimming in (0.5, 1) or <= 0 (0.5 < pi_0 < 1 returned a
negative p-value).
