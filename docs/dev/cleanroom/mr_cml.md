# Clean-room spec: `sp.mr_cml` (MR-cML-BIC / cML-MA-BIC)

Scope: the whole module `src/statspai/mendelian/frontier/cml.py` (audit row
H16, `docs/dev/2026-10-20-license-provenance-audit.md`): `mr_cml`, the
fixed-K solver `_fit_fixed_k` and `MRcMLResult.summary`.

Reference used as a black box: R `MendelianRandomization` 0.10.0
(GPL-2 | GPL-3), function `mr_cML` (help page and outputs only).

## Sources read / not read

Read:

- Xue, Shen & Pan (2021), `xue2021constrained`, AJHG 108(7) 1251-1269,
  full text on PMC (PMC8322939): Material and methods (Equations 1-16:
  model, constrained likelihood, BIC, model averaging, the coordinate-descent
  algorithm with its default start, the Fisher information of Equation 8,
  data perturbation and the GOF tests).
- `?mr_cML` (MendelianRandomization 0.10.0 help page via `tools::Rd2txt`).
- Outputs of `mr_cML` already committed in
  `tests/reference_parity/_fixtures/mr_R.json` (keys `cml_per_k`, `cml_bic`,
  `cml_ma`; generator `tests/reference_parity/_fixtures/_generate_mr_R.R`).
- The repository's tests (`tests/test_mr_frontier.py`,
  `tests/reference_parity/test_mr_cml_parity.py`,
  `tests/reference_parity/test_mr_R_parity.py`), the registry entry and the
  agent card.
- The redacted view of `cml.py` (every function body removed) and the
  outputs of the previous implementation on 101 input configurations
  (18 data sets x up to 5 option sets = 88, plus 13 edge cases), run as a
  black box before it was deleted.

Not read: any MendelianRandomization source (no function bodies printed, no
CRAN/GitHub source), the previous bodies in `cml.py` or their git history.

## Model and estimator (paper, Equations 1-4)

`bx_i ~ N(b_i, sx_i^2)`, `by_i ~ N(theta b_i + r_i, sy_i^2)`, independent
across the `m` variants. Log-likelihood (Eq. 2)

    L = -1/2 sum_i [ (bx_i - b_i)^2 / sx_i^2 + (by_i - theta b_i - r_i)^2 / sy_i^2 ].

For fixed `K` maximise `L` subject to `#{i : r_i != 0} = K` (Eq. 3). Coordinate
descent from `theta = 0`, `b_i = 0` (the paper's default start), iteration `k`:

1. order `(by_i - theta b_i)^2 / sy_i^2` decreasingly; the `K` largest get
   `r_i = by_i - theta b_i`, the rest `r_i = 0`;
2. `b_i = (bx_i / sx_i^2 + theta (by_i - r_i) / sy_i^2) / (1 / sx_i^2 + theta^2 / sy_i^2)` (Eq. 6, with the current theta);
3. `theta = sum_i (by_i - r_i) b_i / sy_i^2 / sum_i b_i^2 / sy_i^2` (Eq. 7, sum over all `i`).

Stop when `|theta_new - theta_old| < tol` (default `1e-7`) or after `max_iter`
(default 100) rounds. Invalid set = `{i : r_i != 0}`.

`BIC(K) = -2 L(K) + K log n` (Eq. 4), `L(K)` the log-likelihood at the final
iterate. `n` is the GWAS sample size (paper: `min(n1, n2)`).

## Standard error (Eq. 8)

Observed information of `(theta, b_valid)` from `-L` with the invalid
variants profiled out (they do not involve theta once `r_i = by_i - theta b_i`):

- `I_tt = sum_valid b_i^2 / sy_i^2`
- `I_tb_i = (2 theta b_i - by_i) / sy_i^2`
- `I_bb = diag(1 / sx_i^2 + theta^2 / sy_i^2)`

`SE = sqrt([I^-1]_11) = (I_tt - sum_valid I_tb_i^2 / I_bb_i)^(-1/2)`.
(The paper writes `(I^-1)_11` for the variance; the reference's numbers are
its square root.)

## Selection and model averaging

- cML-BIC: `K_hat = argmin_K BIC(K)`; report `theta(K_hat)`, `SE(K_hat)`.
- cML-MA-BIC (Eq. 5, Buckland et al.): `w_K ∝ exp(-BIC(K)/2)`,
  `theta_w = sum_K w_K theta(K)`, and

      SE_w = sum_K w_K sqrt(SE(K)^2 + (theta(K) - theta_w)^2)

  i.e. the weights multiply the square roots. The PMC rendering of Eq. 5 is
  ambiguous about where the root sits; the black box settles it: this form
  reproduces `cml_ma$se` to 4e-16, the form with the root outside the sum is
  off by 3.7e-3.
- Inference: normal, `p = 2 Phi(-|est / se|)`, CI `est -/+ z_{1-alpha/2} se`.

## Black-box conventions established

| Convention | How | Result |
| --- | --- | --- |
| start `theta = 0, b = 0`, steps 1-2-3 in the paper's order, Eq. 7 summed over all variants, stop on `|d theta| < 1e-7` | `cml_per_k`, K = 0..6 | estimates and SEs within 1.6e-15 relative, invalid sets identical |
| SE is the square root of `[I^-1]_11` | same | 1.6e-15 |
| BIC with `log(n)` and `n = 17723` selects K = 2 | `cml_bic` | identical estimate, SE, invalid set (12, 14) |
| MA SE has the root inside the weighted sum | `cml_ma` | 4e-16 |
| candidate set `{0, ..., m - 2}` | help page `K_vec` default | default `K_max = m - 2` |

## Defaults and edge cases (StatsPAI API, unchanged unless noted)

- `K_max=None` means `m - 2`; values outside `[0, m - 1]` raise `ValueError`
  naming `K_max` (the previous bound; `m - 1` keeps a single valid variant).
- `n=None`: `UserWarning`, BIC uses `log(m)` (previous StatsPAI behaviour; the
  reference requires `n`).
- Variants with `bx < 0` are sign-flipped together with `by` (the likelihood
  and the algorithm are invariant to this).
- New: non-finite inputs, non-positive standard errors, `n <= 1` and
  `alpha` outside `(0, 1)` raise `ValueError` (the previous code raised a
  `KeyError` for NaN, silently squared negative SEs and took `log(0)`).
- New: `path` gains a `converged` column; a `RuntimeWarning` is issued when
  the coordinate descent hits `max_iter` at the selected K (or, with
  `model_average=True`, at any K with BIC weight >= 1e-3). On the LDL data
  K >= 16 do not reach `1e-7` in 100 rounds; they carry no weight.
- Not implemented: data perturbation (cML-*-DP), GOF tests, random starts.

## Log-likelihood reported

`loglik` (and `path['loglik']`, from which BIC is built) is Equation 2 plus
the Gaussian normalising constant `-1/2 sum_i log(4 pi^2 sx_i^2 sy_i^2)`,
i.e. the full log-density of the `2m` statistics, evaluated after a final
step 1 at the last theta. Both choices were read off the previous
implementation's outputs (the constant is the exact offset, 200.977 on the
LDL data; the final step 1 reproduces its values at K that hit `max_iter`).
BIC differences across K, hence `K_selected`, do not depend on either.

## Comparison with the previous implementation

On the 88 snapshot fits: estimate, SE, CI, p-value and `loglik` agree to
1.1e-14 relative, `K_selected` and the invalid sets are identical, and every
path row agrees (estimate exactly, SE 2e-16, log-likelihood 5e-10, the last
only at K that stop at `max_iter`). Differences are confined to:

- the warning text when `n` is missing (still a `UserWarning` mentioning `n`);
- the new `converged` column of `path` and the new non-convergence warning;
- invalid inputs now raising `ValueError` (13 edge cases: NaN input was a
  `KeyError`; negative SEs were squared silently; `n = 0` gave `log(0)`;
  `alpha = 1.5` was accepted);
- `summary()` wording.
