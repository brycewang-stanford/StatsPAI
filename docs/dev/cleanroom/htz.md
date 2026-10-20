# Clean-room spec: CR2 / HTZ Wald test (`fast/inference.py`)

Scope: audit row H19 (`docs/dev/2026-10-20-license-provenance-audit.md`):
`_htz_per_cluster_quantities`, `_htz_eta_from_quantities`,
`cluster_dof_wald_htz`, `cluster_wald_htz` and the comment block above
`WaldTestResult`, all in `src/statspai/fast/inference.py`. Their bodies and
docstrings were deleted unread and replaced. `WaldTestResult` and the rest
of the file (CRVE, wild bootstrap, Bell-McCaffrey DOF helpers) are outside
this row and were not touched. The audit's M13 helper (`_matrix_power_sym`)
is in `inference/jackknife.py`, not here; it was handled in the MEDIUM-row
pass (`docs/dev/cleanroom/medium_c_rows.md`).

Reference used as a black box: R `clubSandwich` 0.6.2 (GPL-3), public API
only (`vcovCR(type = "CR2")`, `Wald_test(test = "HTZ", null_constant =)`).

## Sources read / not read

Read:

- Pustejovsky & Tipton (2018) (`pustejovsky2018small`), arXiv v2 of
  1601.01981: Section 2 (CR2 adjustment matrices, eq. 5-8, Theorems 1-2) and
  Section 3 (eq. 10-13, the AHT test). Bell & McCaffrey (2002)
  (`bell2002bias`) through eq. (11) of that paper.
- `?Wald_test` (clubSandwich help page).
- The tests and fixtures that call the functions (`tests/test_fast_htz.py`,
  `tests/fixtures/_gen_htz_fixture.R` and its README). Two of these called
  clubSandwich internals: the fixture generator and the live-R test passed
  the contrast through `get_GH`, `get_P_array(all_terms = TRUE)`,
  `matrix_power(Omega, -1/2)` and `total_variance_mat`, with
  `Omega = apply(P, 1:2, function(x) sum(diag(x)))` and
  `nu = q (q + 1) / sum(VM)`. Reading those R lines told me the order of
  operations of the reference at the level of function names; it matches
  the paper's eq. (13) with the expectation of the CR2 estimate in place of
  `G`. Both scripts now use only `Wald_test`'s public output
  (`eta = df_denom + q - 1`); the fixture regenerated from the public API
  differs from the committed one only in the 14th digit.
- The redacted view of `fast/inference.py` (signatures only), including the
  signatures of the shared validators `_factorize_cluster_labels` and
  `_prepare_weights`, which the new code calls.
- Outputs of the previous implementation on 40 random designs (q from 1 to
  5, unbalanced and singleton clusters, cluster-level regressors, non-zero
  nulls), recorded before the bodies were deleted.

Not read: any clubSandwich source; the old bodies and docstrings; the old
design note `docs/superpowers/specs/2026-04-27-htz-clubsandwich-parity-design.md`
(marked superseded at its top without being opened; deleted on 2026-10-21);
the implementation plan `docs/superpowers/plans/2026-04-27-htz-clubsandwich-parity.md`
(deleted on 2026-10-21 after reading only its title).

## Specification

OLS with `M = (X'X)^{-1}` (or the given `bread`), `H = X M X'`, working
model `Phi = I`, `W = I` (equal weights are allowed; unequal weights raise
`NotImplementedError`, as before).

1. CR2 adjustment per cluster: `A_g = (I - H_gg)^{+1/2}`, the symmetric
   square root of the Moore-Penrose inverse (eq. 7-8); eigenvalues at or
   below `n_g * eps * max(1, largest)` count as zero.
2. `V_R = sum_g (R M X_g' A_g e_g)(R M X_g' A_g e_g)'`.
3. `Omega` = expectation of `V_R` under the working model,
   `sum_g R M X_g' A_g (I - H_gg) A_g X_g M R'` (equals `R M R'` when the
   CR2 estimator is exactly unbiased, Theorem 1).
4. `g_s` = columns of `Omega^{-1/2}`; `p_si = (I - H)_i' A_i X_i M R' g_s`;
   `eta = q (q + 1) / sum_{s,t} sum_{i,j} (p_si'p_tj p_ti'p_sj +
   p_si'p_sj p_ti'p_tj)` (eq. 13, all cluster pairs). Computed without
   `n`-vectors: `p_si'p_tj = 1{i = j} u_si'u_tj - (X_i'u_si)' M (X_j'u_tj)`
   with `u_si = A_i X_i M R' g_s`, because `I - H` is symmetric idempotent.
5. `Q = (R beta - r)' V_R^{-1} (R beta - r)`, `F = (eta - q + 1)/(eta q) Q`,
   `p = P(F(q, eta - q + 1) > F)` (eq. 12). `Q = 0` exactly gives `p = 1`.

Edge behaviour (kept from the previous implementation's observable
behaviour or its tests): `R` must have `k` columns, at least one row and full
row rank; `G > q`; a singleton cluster warns (`RuntimeWarning`);
`eta <= q - 1` raises `NumericalInstability` (the F reference does not
exist; the previous code raised there too). New: a (near-)singular `V_R`
(condition number above 1e12, e.g. a restriction on a cluster fixed
effect) warns and uses the pseudo-inverse; clubSandwich returns rounding
noise there (one such case was generated and dropped from the fixture).

## Evidence

- `tests/reference_parity/test_htz_blackbox_R_parity.py` against
  `htz_blackbox_R.json` (generator `_generate_htz_blackbox_R.R`, data
  `_generate_htz_blackbox_data.py`): five designs including the two-way
  dummy panel whose `I - H_gg` is singular, q = 1 to 4, non-zero nulls;
  `eta`, F, p and `V_R` agree to 3.4e-14 (tolerance 1e-8 / 1e-7). Also the
  identity that at `q = 1`, `eta` is the Satterthwaite DOF of eq. (11).
- `tests/test_fast_htz.py` (24 tests; the frozen fixture and the live-R test
  now through the public API).
- The 40-design comparison with the previous implementation: `eta`, F, Q and
  `V_R` to 6e-14, p to 3e-12; the same error on the degenerate design.
