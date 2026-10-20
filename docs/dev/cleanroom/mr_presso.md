# Clean-room spec: MR-PRESSO (`sp.mr_presso`)

Scope: `mr_presso` and its nested helpers in
`src/statspai/mendelian/diagnostics.py` (moved unchanged to
`src/statspai/mendelian/presso.py` on 2026-10-20; the old import path still
works) (audit row H15,
`docs/dev/2026-10-20-license-provenance-audit.md`). The other functions in
that file (`mr_heterogeneity`, `mr_steiger`, `mr_radial`, ...) were not
flagged and are unchanged. Public API kept: the signature
`mr_presso(bx, by, sx, sy, *, n_boot=1000, sig_threshold=0.05, seed=None)`
and every `MRPressoResult` field.

Reference used as a black box: R `MRPRESSO` 1.0 (GPL-3), exported function
`mr_presso` (help page and outputs only).

## Sources read / not read

Read:

- Verbanck, Chen, Neale & Do (2018), `verbanck2018detection`, Online
  Methods ("MR-PRESSO global test", "outlier test", "distortion test"), from
  the PMC author manuscript PMC6083837 (JATS XML with MathML via NCBI
  E-utilities; the BioC text for the prose).
- `?mr_presso`, `?MRPRESSO-package` (MRPRESSO 1.0) via `tools::Rd2txt`.
- Outputs of `MRPRESSO::mr_presso` on the committed `mr_ldl.csv` (the
  existing fixture `mr_R.json` and a K = 2000 run) and on the new fixture
  `tests/reference_parity/_fixtures/mr_presso_R.json` (3 scenarios x 6
  seeds, K = 1000), written by `_generate_mr_presso_R.R`.
- The repository's tests (`tests/test_mr_diagnostics.py`,
  `tests/test_correctness_v150.py`, `tests/reference_parity/test_mr_parity.py`,
  `tests/reference_parity/test_mr_R_parity.py`, `_generate_mr_R.R`), the
  registry entry, and the other functions of `diagnostics.py`.
- The redacted view of `diagnostics.py` (body of `mr_presso` and its nested
  helpers removed; the `MRPressoResult` class, which only lists fields and a
  summary, read normally) and the outputs of the old implementation on 24
  configurations, run as a black box before it was deleted.

Not read: any MRPRESSO R source (no function bodies printed, no GitHub
source); the old `mr_presso` body or its git history.

## Method (paper, single exposure)

Notation: `bx_j, by_j, sx_j, sy_j`, `w_j = 1/sy_j^2`, `J` variants.

- **IVW fit** (`_ivw_regression`): weighted no-intercept regression of `by`
  on `bx`; slope `sum w bx by / sum w bx^2`; SE `sqrt(s2 / sum w bx^2)` with
  `s2 = sum w r^2 / (J - 1)` (residual-scaled, not floored); t test on
  `J - 1` df. Used for the raw and the outlier-corrected estimates.
- **Leave-one-out slopes** (`_loo_slopes`): `b_{-j} = (S_xy - w_j bx_j by_j)
  / (S_xx - w_j bx_j^2)`.
- **Observed RSS** (`_presso_rss`): `RSS_obs(j) = (by_j - b_{-j} bx_j)^2`;
  global `RSS_obs = sum_j w_j RSS_obs(j)`.
- **Null draws** (K = `n_boot`): `bx*_j ~ N(bx_j, sx_j^2)`,
  `by*_j ~ N(b_{-j} bx_j, sy_j^2)`; on each draw the leave-one-out slopes are
  recomputed from `(bx*, by*)` and `RSS*(j) = w_j (by*_j - b*_{-j} bx*_j)^2`.
- **Global p** = `#{k: sum_j RSS*_k(j) > RSS_obs} / K` (strict `>`, paper
  step 4; `k / K`, so 0 means "below 1/K").
- **Outlier test**, only when global p < `sig_threshold`:
  `p_j = min(1, J #{k: RSS*_k(j) > w_j RSS_obs(j)} / K)` (Bonferroni);
  outliers are `p_j < sig_threshold`. A warning fires when
  `n_boot * sig_threshold < J` (one exceeding draw already gives adjusted
  p = J/K).
- **Corrected estimate**: IVW on the non-outliers (needs >= 2 left, else a
  warning and `None`).
- **Distortion**: `D_obs = 100 (b_raw - b_cor) / |b_cor|`. Null (our reading
  of the paper's prose): replace the `n_out` outliers with `n_out` variants
  drawn with replacement from the non-outliers, so `J` variants remain;
  `b_null` = IVW slope of that set on the observed data;
  `D_null = 100 (b_raw - b_null) / |b_null|`; p = share of draws with
  `|D_null| > |D_obs|`. Drawn after the global-test draws from the same
  generator.

## Conventions established from the black box

| Convention | How established |
| --- | --- |
| Global RSS weighted by `1/sy^2`, per-variant table unweighted | `RSSobs` 102.7158 (ldl) equals the weighted sum; the outlier table's `RSSobs` equals the unweighted squares (both within 1e-10, observed ~1e-15) |
| IVW SE residual-scaled without the max(1, .) floor; t-based p on J-1 df | Main MR results Sd / P-value of raw and corrected fits, within 1e-10 |
| Null outcome centred at `b_{-j} bx_j`, LOO slopes refitted on each draw | `sim_clean` global p: R 0.342 (6 x 1000 draws, MC SE 0.006); refitted slopes 0.330, observed slopes 0.248, full-sample slope 0.199 (2e5 draws each) |
| Outlier test only when the global test rejects | `sim_clean` returns only "Global Test" |
| p-values `k / K`; R prints `<1/K` for k = 0 | help page ("precision ... 1/NbDistribution") and outputs |
| Bonferroni factor `J`, capped at 1 | outlier table values (1 for most variants; J/K multiples) |

## Results against the fixture

- T2 (rtol 1e-10, `tests/reference_parity/test_mr_presso_R_parity.py`): raw
  and corrected estimate / SE / p, global and per-variant observed RSS,
  outlier set (identical in all 6 reference runs and all 6 StatsPAI runs),
  distortion coefficient; observed agreement 1e-15.
- S (stochastic screen, not a parity claim): global p means (ldl 0 / 0,
  sim_out 0 / 0, sim_clean 0.319 vs 0.342) and per-variant exceedance
  shares, tolerance 4 MC SE of the difference.
- **Open item, C (unexplained): the distortion p-value.** StatsPAI
  (6 x 1000 draws) 0.605 (ldl), 0.801 (sim_out); MRPRESSO 0.519, 0.926.
  Over a hundred combinations of readings of the paper's description were
  tried (six resampling schemes, observed vs simulated data, five
  normalisations of D, signed vs absolute comparison); none reproduces both
  reference values within Monte Carlo error. The old
  StatsPAI implementation did not reproduce them either (ldl 0.59-0.65).
  Not compared in the test; documented in its module docstring.

## Old vs new (StatsPAI before / after, 24 configurations)

- Deterministic outputs (raw fit, RSS_obs, corrected fit when the outlier
  set agrees, distortion coefficient): equal to 1.4e-14.
- Global p-values: identical in 23 of 24 configurations (the new draws
  happen to consume the generator in the same order), the remaining one
  (`neg`, B = 2000) 0.7055 vs 0.6965.
- Outlier sets: identical in 22 of 24; the two differences are B = 200 runs
  with `sig_threshold = 0.1`, where variants sit at the Monte Carlo
  resolution (one variant dropped / added).
- Distortion p: same construction class, different draws (differences up to
  0.07 at B = 200).
- "cannot resolve" warning: same condition, reworded text.
