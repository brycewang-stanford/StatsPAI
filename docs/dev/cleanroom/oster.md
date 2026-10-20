# Clean-room specification: exact Oster (2019) delta and beta*

Audit row M1 of `docs/dev/2026-10-20-license-provenance-audit.md`:
`diagnostics/_oster.py` described itself as "a port of the Mata code of
Oster's Stata `psacalc` 2.1 (functions `bound`, `d1quadsol` and
`dnot1cubsol`)". `psacalc` states no licence. The bodies of
`oster_delta_exact` and `oster_beta_exact` were deleted unread and rewritten.
The input builders (`oster_inputs`, `oster_inputs_fe`, `resolve_r_max`) and
the first-order approximations were not part of the flagged port and are
unchanged.

## Sources (and only these)

- Oster (2019) `[@oster2019unobservable]`: the estimator, delta, R_max and
  assumption 3 (the bias has the sign of the observed coefficient movement).
- R package `robomit` 1.0.7 (Sergei Schaub; MIT, `THIRD_PARTY_NOTICES.md`):
  the source of `o_beta` and `o_delta` was read, as the brief allows for a
  permissively licensed implementation of the same method. It states that
  it "is based on the psacalc command in Stata".
- Black-box outputs: Stata `psacalc` 2.1 fixtures already in the repository
  (`tests/reference_parity/test_inference_sens_stata_parity.py`,
  `test_oster_absorb_Stata_parity.py`), and the previous StatsPAI functions
  called on 4,000 random input sets (delta in {1, 0, 0.5, 1.5, 2, 3.7, -1,
  uniform}, R_max above and below R_t) and 12,000 delta* evaluations.

## Notation

`b_o, R_o`: short regression; `b_t, R_t`: long regression; `s_yy`, `s_xx`,
`t_x`: the three variances of the module docstring. `gap = b_o - b_t`,
`r_move = (R_t - R_o) s_yy`, `r_room = (R_max - R_t) s_yy`. The unknown is
the bias `nu = b_t - beta*`.

## delta for a target beta (`oster_delta_exact`)

With `nu = b_t - beta`, delta = N / D where

- `N = nu r_move t_x + nu s_xx t_x gap^2 + 2 nu^2 t_x gap s_xx + nu^3 (t_x s_xx - t_x^2)`
- `D = r_room gap s_xx + nu r_room (s_xx - t_x) + nu^2 t_x gap s_xx + nu^3 (t_x s_xx - t_x^2)`

## beta* for a given delta (`oster_beta_exact`)

- delta = 1: `nu` solves a quadratic with
  `mid = r_room (s_xx - t_x) - r_move t_x - s_xx t_x gap^2`,
  discriminant `mid^2 + 4 r_room gap^2 s_xx^2 t_x`, leading term
  `-2 t_x gap s_xx`; the two roots `nu = (-mid -/+ sqrt(disc)) / lead`
  give `beta = b_t - nu`. Choose the root closer to `b_t`; if its bias
  `beta - b_t` does not have the sign of `b_t - b_o`, take the other. A
  negative discriminant gives NaN (no warning beyond NumPy's).
- delta != 1: monic cubic `nu^3 + a nu^2 + b nu + c` with
  `scale = (delta - 1)(t_x s_xx - t_x^2)`,
  `a = t_x gap s_xx (delta - 2) / scale`,
  `b = (delta r_room (s_xx - t_x) - r_move t_x - s_xx t_x gap^2) / scale`,
  `c = r_room delta gap s_xx / scale`. With `q = (a^2 - 3b)/9` and
  `r = (2a^3 - 9ab + 27c)/54`: if `r^2 - q^3 < 0` there are three real
  roots `nu_j = -2 sqrt(q) cos((theta + s_j)/3) - a/3`,
  `theta = arccos(r / sqrt(q^3))`, `s = 0, 2 pi, -2 pi`; otherwise the one
  real root by Cardano's formula with real cube roots.
- Three roots: rank by `(beta - b_t)^2`; a root whose bias has the wrong
  sign gets, in turn, one more than the largest distance so far; report
  the first, the other two as alternatives.
- Return `{"beta", "alternatives", "roots"}` with `roots` in solution order.

## Acceptance

All psacalc fixtures at their registered tolerance (1e-12); the 4,000 /
12,000 recorded previous outputs to <= 6.1e-13 relative (all branches,
including NaN and no-admissible-root cases, agree).
