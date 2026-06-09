# Stability and Validation Tiers

StatsPAI now separates **API lifecycle** from **numerical validation evidence**. This is the main correction to the older catalogue: `stability='stable'` no longer means "R/Stata parity-grade" by itself.

## Three Fields

| Field | Scope | Meaning |
| --- | --- | --- |
| `stability` | whole function API | `stable`, `experimental`, or `deprecated` |
| `validation_status` | evidence for numerical output | `certified`, `validated`, `api_stable`, `experimental`, or `deprecated` |
| `limitations` | parameter/variant gaps | documented unsupported variants inside an otherwise usable function |

Use `stability` when you care about public API compatibility. Use `validation_status` when you care about audited numerical evidence.

Current JSS source-snapshot audit counts: 52 `certified`, 25 `validated`, 940 `api_stable`, and 3 `experimental` registry symbols. The intentionally harsh denominator is that 751 stable auto-registered symbols still lack parity backing; treat them as API-stable, not numerically validated. The audit decomposes that denominator into class-like/function-like and category counts, so breadth remains auditable rather than becoming a hidden validation claim. Within the hand-written stable API surface, the current audit enforces zero unbacked entries: API-only helpers carry unit-contract evidence while remaining `api_stable`, not numerically validated. Unit and regression tests are API-contract evidence; they do not promote a function to `validated` without known-truth, reference-parity, external-parity, coverage, or explicit convention evidence. `Paper-JSS/replication/results/validation_evidence_audit.{json,md}` verifies that all 77 certified/validated symbols have registry-attached evidence notes, that certified symbols carry attached R/Stata parity-module evidence, and that validated symbols are not backed only by unit/regression tests. Package metadata is still `1.16.0`; source-snapshot fixes marked `1.16.0+` should be synchronized with a tagged release before final publication. The JSS archive records this boundary in `Paper-JSS/replication/results/source_snapshot_manifest.{json,md}`, and `cd Paper-JSS && make release-audit` is the strict gate for a clean tagged final-publication snapshot.

## Stability

- `stable`: public signature is locked under SemVer minor releases.
- `experimental`: method/API may shift across minor versions.
- `deprecated`: scheduled for removal; replacement should be documented in `MIGRATION.md`.

## Validation

- `certified`: cross-language or published-reference parity evidence exists, usually from `tests/r_parity/`, `tests/stata_parity/`, or published-replication fixtures.
- `validated`: known-truth, reference-parity, external-parity, coverage, or explicit convention evidence exists, but the function is not in the main Track A R/Stata harness.
- `api_stable`: stable public API. Unit/regression tests may attach API-contract evidence here, but that evidence is not numerical validation.
- `experimental`: mirrors `stability='experimental'`.
- `deprecated`: mirrors `stability='deprecated'`.

## Filtering

```python
import statspai as sp

sp.list_functions()                              # all registered functions
sp.list_functions(stability="stable")            # stable API
sp.list_functions(validation_status="certified") # parity-backed functions
sp.agent_cards(validation_status="certified")    # parity-backed agent cards

spec = sp.describe_function("regress")
spec["stability"]          # "stable"
spec["validation_status"]  # "certified"
spec["validation_notes"]   # parity artifact / reference notes
```

```bash
statspai list --stability experimental
statspai list --validation certified
statspai describe rdrobust
```

`sp.help()` prints both `STABILITY` and `VALIDATION` count blocks. Per-function help shows `Stability:`, `Validation:`, `Evidence:`, and `Known limitations` when available.

## Promotion Path

1. Promote `experimental` to `stable` when the public API is ready for SemVer compatibility.
2. Promote `api_stable` to `validated` when analytic/reference parity tests exist.
3. Promote `validated` to `certified` when the function enters the cross-language or published-reference parity harness.
4. Remove a `limitation` only when the unsupported variant lands with its own test.

## Current Limitation Hotspots

These are machine-readable through `sp.describe_function(name)["limitations"]` and should be treated as the priority backlog for production hardening:

- `callaway_santanna`: repeated cross-sections currently support only `estimator="reg"` with `control_group="nevertreated"`.
- `rdrobust`: observation-level weights are reserved and raise `NotImplementedError`; exact R parity is attached to `bwselect="cct"` or common manual bandwidths, while the default `mserd` selector is a documented convention.
- `rddensity`: native default bandwidths, mass-point ECDF handling, and jackknife CJM local-density inference mirror `rddensity::rddensity` on the JSS parity fixture. Manual side-specific bandwidths are still treated as explicit user controls; `backend="r"` remains available when direct execution of the R package is required.
- `synth`: ADH/Synth parity requires the same `special_predictors` recipe; SDID/augmented/gsynth rows include documented regularisation or local-optimum convention gaps.
- `causal_forest`: the NSW-DW parity row is overlap-diagnostic evidence, not a clean ATT point-estimate parity claim.
- `did_imputation`: parity is aggregation-convention sensitive; inspect `sp.parity_gap_report()` before reporting exact cross-language equality.
- `etwfe`: the default top-level estimate is cohort-share weighted; use `sp.etwfe(..., panel=False, cluster=...)` followed by `sp.etwfe_emfx(..., weighting="treated")` for R `etwfe::emfx(type="simple")` point-estimate and clustered-SE parity.
- `hal_tmle`: `variant="projection"` is reserved and raises `NotImplementedError`.
- `network_exposure`: only `design="bernoulli"` is implemented.
- `etwfe`: `panel=False` with `cgroup="nevertreated"` is not implemented.
- `continuous_did`: `method="cgs"` is an MVP without full CGS parity.
- `did_multiplegt_dyn`: experimental MVP; switch-off events, analytical IF variance, and heteroskedastic weights are not implemented.

## Auditing

```bash
python scripts/stability_audit.py
python scripts/stability_audit.py --unbacked
python scripts/stability_audit.py --check
python scripts/stability_audit.py --json
python Paper-JSS/replication/scripts/validation_evidence_audit.py
```

`scripts/stability_audit.py --check` fails if any hand-written stable API entry lacks attached validation or API/unit-contract evidence. Auto-registered entries are reported separately because they represent breadth imported into the registry, not the validated numerical core defended in the JSS paper.

The JSS packager also extracts Python source paths from registry evidence notes. The current submission manifest includes 133 such registry evidence files, and `Paper-JSS/replication/scripts/verify_submission_package.py` fails if any referenced evidence file is absent from the archive.

Programmatic evidence summaries:

```python
sp.validation_report()
sp.coverage_matrix(level="parity")
sp.parity_gap_report()
```

`sp.parity_gap_report()` parses the already-generated 3-way parity table and reports documented convention gaps, missing Stata siblings, priorities, and next actions.

*Last updated: JSS source-snapshot validation audit (2026-05-31).*
