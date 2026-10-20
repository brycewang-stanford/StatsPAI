# Clean-room spec: binned Gaussian KDE (`core/_binned_kde.py`)

Scope: audit rows H23 (`regression/interflex.py::r_density`) and H24
(`decomposition/_common.py::_bw_nrd0`, `kde_at_dineq`),
`docs/dev/2026-10-20-license-provenance-audit.md`. Both helpers now delegate
to one new module, `src/statspai/core/_binned_kde.py`; their old bodies were
deleted unread.

References used as black boxes: R 4.5.2 `stats::bw.nrd0` and
`stats::density` (R is GPL-2 | GPL-3), and `dineq::rif` 0.1.0 (GPL-3).

## Sources read / not read

Read:

- R help pages `?density` and `?bw.nrd0` (via `tools::Rd2txt`), and
  `?dineq::rif`.
- Silverman (1986), eq. (3.31), as cited by `?bw.nrd0`
  (`silverman1986density`); Silverman (1982), Algorithm AS 176
  (`silverman1982algorithm`); Wand (1994) on linear binning
  (`wand1994fast`).
- Outputs of the reference functions on the data sets of
  `tests/reference_parity/_fixtures/_generate_binned_kde_blackbox_R.R` and
  on throw-away probes (zero weights, points outside the grid).
- The redacted views of `interflex.py` and `decomposition/_common.py`
  (bodies and docstrings of the two contaminated helpers removed).
- Outputs of the previous StatsPAI helpers on 63 random inputs, recorded
  before their bodies were deleted, to confirm that the rewrite keeps every
  output (agreement 1e-12 or better, except the single-observation case
  below).

Seen by accident (recorded for completeness): while grepping for callers, one
comment line of the old `r_density` body ("linear binning with unit total
mass onto n points, padded to 2n") and one line of `kde_at_dineq`
(`h = max(_bw_nrd0(y_pos), 1e-12)`). Both describe the generic method in the
literature; the grid size and the treatment of weights were then settled
from R's outputs (below), not from these lines.

Not read: any R source (`density.default`, `bw.nrd0`, `BinDist`, `dineq`).

## Specification

Bandwidth (`silverman_bandwidth`): `0.9 * s * n^(-1/5)`, `s = min(sd,
IQR/1.34)`, `sd` with divisor `n - 1`, IQR from type-7 quartiles. `?bw.nrd0`
only says that a positive result is guaranteed when the quartiles coincide;
black-box runs fix the fall-backs: `s = sd` when the IQR is zero
(`c(-20, 0 x 98, 20)`), `|x[1]|` when the data are constant (`rep(3, 10)`
gives `0.9 * 3 * 10^-0.2`), and 1 when that is zero (`rep(0, 5)`). R refuses
a single observation; StatsPAI takes `s` from the fall-backs (`|x|` or 1).

Density (`binned_gaussian_kde`): output points `linspace(from, to, n)` with
`from = min(x) - cut*bw`, `to = max(x) + cut*bw` (`?density`). The binning
grid spans `[from - ext*bw, to + ext*bw]` (`ext = 4`) with
`M = max(n, 512)` points, rounded up to a power of two above 512
(`?density`, argument `n`). Black-box candidates for the grid: `M = 2n`
points, spacing `(hi - lo)/M`, and `M = n` points with spacing
`(hi - lo)/(M - 1)`; only the last matches R (1e-15 against 1e-4 for the
others). Each observation's mass is split linearly between its two
neighbouring grid points; mass outside the grid is dropped (confirmed: a
point evaluation ignores observations farther than `ext*bw`). The binned
masses are convolved with `phi(offset/bw)/bw` at every grid offset (zero
padded FFT), the result is linearly interpolated onto the output points and
negative rounding noise is clipped at zero.

Weights (`weights=`): used as given (R's documented default is `1/n` each);
they do not enter the bandwidth (`?density`: "weights are not taken into
account for automatic bandwidth rules").

Density at a point (`binned_density_at`): the same estimator with
`from = to = point`. `dineq::rif(method = "quantile")` returns
`q + (tau - 1{y < q}) / f(q)`; recovering `f(q)` from its output on five data
sets (negative values, weights, ties at zero) matches this with masses
`w / sum(w)` and the bandwidth from all observations, to 1e-15.

Zero weights: `dineq` keeps observations with weight 0 in the bandwidth (and
returns `NA` for their RIF). StatsPAI drops them before computing anything,
as the previous implementation did; with no zero weights the two agree. This
departure is documented in `kde_at_dineq`'s docstring.

## Evidence

- `tests/reference_parity/test_binned_kde_blackbox.py` against
  `binned_kde_blackbox_R.json`: bandwidth 1e-13, densities 1e-12 relative to
  the peak (observed about 1e-14) for nine data sets x six settings
  (`n = 100 / 512 / 1000`, `cut = 1`, fixed `bw`, weights), and `dineq`'s RIF
  density and full RIF vector to 1e-12.
- Track A `87_interflex` and `32_rif` reproduce their committed results (see
  the commit that re-records their traces).
