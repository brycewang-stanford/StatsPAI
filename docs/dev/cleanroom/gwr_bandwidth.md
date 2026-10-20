# Clean-room spec: GWR bandwidth search (`spatial/gwr/bandwidth.py`)

Scope: `src/statspai/spatial/gwr/bandwidth.py` in full (audit row H21,
`docs/dev/2026-10-20-license-provenance-audit.md`): the flagged golden-section
helper and, because the module docstring described the whole file as a
reproduction of `GWmodel`, also `gwr_bandwidth`, the criterion helper and
the distance helper. The file was replaced; the old one was not opened.

Reference used as a black box: R `GWmodel` 2.4-1 (GPL >= 2), exported
functions `bw.gwr` and `gold` only.

## Sources read / not read

Read:

- Brunsdon, Fotheringham & Charlton (1996) (`brunsdon1996geographically`)
  and Gollini et al. (2015) (`gollini2015gwmodel`) for GWR and its
  bandwidth selection by CV / AICc; Kiefer (1953) (`kiefer1953sequential`)
  for golden-section search. All three verified via Crossref and doi.org and
  added to `paper.bib`.
- `?bw.gwr` (GWmodel help page). It documents the arguments and kernels but
  not the search interval or the rounding of adaptive bandwidths.
- `args(GWmodel::gold)` (formal arguments only: `fun, xL, xU, adapt.bw`).
- Black-box runs: `bw.gwr`'s printed trace (each evaluated bandwidth and
  score) on Georgia for 8 settings, and `gold` called with objectives that
  record their argument (40 integer and 24 continuous searches).
- The redacted view of the old module (signatures only) and its callers and
  tests. `tests/reference_parity/test_spatial_survey_R_parity.py` states
  "bounds, floor / round integer probes, 1e-4 stop".
- Outputs of the previous implementation on 240 configurations (two data
  sets, five kernel names, adaptive / fixed, four criteria, three interval
  settings), recorded before the file was replaced, and on 180 small
  configurations (`n` from 8 to 20), run as a black box from the main
  worktree's unmodified copy.
- Seen after the rewrite, in the diff of the regenerated schema
  (`schemas/functions.json`, which carries the old parameter descriptions):
  the old docstring documented the adaptive interval as `[20, n]`, or
  `[k + 2, n]` when `n <= 20`, and `tol` as applied both to the step and to
  the criterion difference. The second point had already been established
  from the traces; the first was then confirmed on the 180 small
  configurations (`k` = local coefficients including the constant: 178 of
  180 match with `k + 2`, the other two once infeasible probes score +inf).

Not read: any `GWmodel` source (R or C++), and `GWmodel:::gold`'s body.

## Specification (established from the black-box runs)

Search interval: adaptive `[20, n]` neighbours; fixed `[u / 5000, u]` with
`u` the largest Euclidean distance between two points (Georgia: the first
two probes 345463.8 and 213551.1 imply `[111.87, 558903.0]`, and
`d_max / 5000 = 111.78`). `bw_min` / `bw_max` (StatsPAI options) replace the
ends; a given `bw_max` also sets the default lower end `bw_max / 5000`
(previous behaviour, kept).

With `n <= 20` the adaptive interval `[20, n]` is empty; the search then
starts at `k + 2` (`k` local coefficients, the constant included), the
previous StatsPAI behaviour (`GWmodel` has no rule for it).

Search (`_golden_section_search`), `r = (sqrt(5) - 1) / 2`:

1. `d = r (upper - lower)`; upper point `lower + d`, lower point
   `upper - d`; both evaluated (upper first).
2. While `|f_upper - f_lower| > tol` and `d > tol`: `d <- r d`; if
   `f_upper < f_lower` the lower end moves to the lower point, the upper
   point becomes the lower point and a new upper point `lower + d` is
   evaluated; otherwise symmetrically with a new lower point `upper - d`.
3. Result: the upper point if its value is strictly smaller, else the lower
   point.

Integer searches (adaptive): the first upper point is rounded down and the
first lower point to the nearest integer; later upper points are rounded to
the nearest integer and later lower points down; the integers are what is
kept. A brute-force search over rounding rules (floor / ceiling / round /
truncation for each of the four positions, rounded or raw storage, step
recomputed from the bracket or shrunk geometrically) found this the only
family that reproduces all 40 integer traces; recomputing the step from the
bracket reproduced 5 of 40. Consequences that the reference shares: an
adaptive search can return a bandwidth one or two neighbours outside the
interval, and on objectives with no ties it stops through the `d > tol`
condition after about `log(tol / range) / log(r)` steps.

Continuous searches (fixed): the same steps without rounding; all 24 traces
agree bitwise (17-digit fixture).

Infeasible probes: a bandwidth with no GWR fit (one neighbour or less,
or a negative count after an integer search drifts below its lower end)
scores `+inf` and a `RuntimeWarning` reports how many probes failed; if
every probe fails, the first error is raised. The previous implementation
also scored them as unusable (its outputs on the two small configurations
where this happens are reproduced), but silently.

Criteria: AICc, AIC, BIC and CV (leave-one-out residual sum of squares) of
`sp.gwr`; `GWmodel` offers AICc and CV.

Behaviour change: an unknown kernel name now raises `MethodIncompatibility`.
The previous code returned a bandwidth for, e.g., `kernel='tricube'` (the
first lower probe), which is what a search over undefined criterion values
returns.

## Evidence

- `tests/reference_parity/test_gwr_golden_search_blackbox.py`: the 64
  recorded searches probe for probe; `bw.gwr` on simulated data and Georgia
  for three kernels x adaptive / fixed x AICc / CV (adaptive exact, fixed
  1e-10).
- `test_spatial_survey_R_parity.py::test_gwr_bandwidth_matches_bw_gwr` (the
  older 8-setting fixture) and `tests/spatial/test_gwr.py`.
- The 240-configuration comparison with the previous implementation: every
  valid configuration bitwise equal.
