# Reproducibility & numerical stability

A result you cannot reproduce is a result you cannot trust. StatsPAI treats
numerical reproducibility as a first-class contract, defended by tests and made
explicit here so you know exactly what is guaranteed across versions — and what
the process is when a number *does* legitimately change.

## The guarantee

> **Within a minor-version line, a flagship estimator returns the same headline
> numbers on the same data — bit-for-bit up to floating-point last-bit noise —
> unless a change is documented as a correctness fix.**

Concretely:

- **Across versions (self-consistency).** Flagship headline outputs are pinned
  in a **version golden master** (`tests/golden_master/`). Every CI run recomputes
  them on fixed-seed data and fails if any digit drifts beyond
  `rtol=1e-6 / atol=1e-8`. So an upgrade cannot silently move your point
  estimate or SE.
- **Against R / Stata (parity).** Estimators with a reference implementation are
  pinned to frozen R/Stata outputs in `tests/reference_parity/` /
  `tests/stata_parity/`, at tolerances documented in
  `tests/reference_parity/REFERENCES.md` (machine-precision `1e-6`–`1e-9` for the
  bit-exact tier; documented looser tolerances for iterative estimators). The R
  package versions behind those fixtures are recorded in
  [`tests/reference_parity/R_PACKAGE_VERSIONS.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/reference_parity/R_PACKAGE_VERSIONS.md)
  so upstream drift is visible, not silent.
- **Per-function evidence.** `sp.parity_status("feols")` reports what a given
  function was validated against and to what tolerance — the auditable map of
  where parity holds and where it is `unverified`.

## What is *not* frozen

- **Last-bit floating point** across CPUs / BLAS builds. Tolerances absorb this.
- **`experimental` functions** (`sp.list_functions(stability="experimental")`)
  may still shift; they are excluded from the cross-version guarantee.
- **Rendering / formatting** (table whitespace, plot styling) — guarded
  separately by `tests/output_snapshots/`, not by numerical tolerance.
- **Reported wall-clock.** Performance changes (see
  [performance](performance.md)) never change the numbers.

## When a number legitimately changes (the correctness-fix process)

Sometimes a number *should* move — a bug is fixed, or a convention is corrected.
That is allowed, but never silently (CLAUDE.md §12):

1. **Document it.** Add a `⚠️ Correctness` entry to `CHANGELOG.md` and a
   migration note to `MIGRATION.md` explaining what moved and why.
2. **Re-pin the golden master.** After confirming the new value is correct, run

   ```bash
   STATSPAI_UPDATE_GOLDEN=1 pytest tests/golden_master/
   ```

   to make the corrected number the new baseline.
3. **Never re-pin to silence a red build** you do not understand — investigate
   *why* the number moved first. The whole point of the gate is to make an
   unexplained move impossible to merge by reflex.

## Reproducing a result yourself

```python
import statspai as sp

# Pin the package version your analysis was run under:
sp.__version__                      # record this in your paper / replication pack

# Check what a function was validated against:
sp.parity_status("rdrobust")        # status / reference / tolerance
```

For a full replication bundle (txt + md + tex + xlsx + figures in one call) see
`sp.cs_report(..., save_to=...)` and the
[replication workflow](replication_workflow.md) guide.
