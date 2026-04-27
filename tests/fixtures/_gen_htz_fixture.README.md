# HTZ clubSandwich fixture

These files are the **frozen reference output** from R clubSandwich's
`Wald_test(test="HTZ")` — used by `tests/test_fast_htz.py::test_htz_frozen_fixture_matches_clubsandwich`
to lock numerical parity to `rtol < 1e-8` on every CI run, including
environments without R installed.

## Files

- `_gen_htz_fixture.R` — generator script (developer-only, NOT in CI).
- `htz_panel_{q1,q2,q3_unbal}.csv` — actual panel data R simulated.
- `htz_clubsandwich.json` — `Wald_test(test="HTZ")` outputs (η, F, p, Q, V_R)
  for each panel.

## Regeneration

```bash
cd tests/fixtures && Rscript _gen_htz_fixture.R
```

## Pinned versions

When this fixture was last regenerated:

- R version: 4.5.2
- clubSandwich: 0.6.2

If a future clubSandwich release changes the HTZ formula (unlikely — the
algorithm is from a 2018 paper), regenerate AND audit every `eta` /
`F_stat` / `p_value` change.

## Why CSV + JSON, not RNG sync

`numpy.random` and R's `set.seed()` produce different streams. The CSV
captures the actual numerical panel R generated, so the Python test reads
byte-identical input. Trying to sync RNGs across languages is a known
foot-gun.
