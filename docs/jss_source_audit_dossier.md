# JSS Source-Audit Dossier

This dossier is the package-facing source audit shipped with the JSS
submission archive. It is not a blanket validation claim: StatsPAI uses
`validation_status` to separate certified or validated numerical evidence from
API-stable breadth.

For the paper-specific reviewer path, start with `Paper-JSS/README.md` in the
submission archive. That path runs Tier 1 without live R or Stata and points to
the exact audit artifacts under `Paper-JSS/replication/results/`.

## Project Status

- Repository: <https://github.com/brycewang-stanford/StatsPAI>
- Package archive: <https://doi.org/10.5281/zenodo.19933900>
- PyPI: <https://pypi.org/project/StatsPAI/>
- License: MIT, with a plain-text `LICENSE` file in the repository.
- Package metadata version: `1.16.0`, released on 2026-05-29.
- JSS source-snapshot audit date: 2026-05-31.
- Public GitHub repository creation date: 2025-07-26.

## Software Scope

StatsPAI exposes a unified Python interface for causal inference and applied
econometrics. The live registry reports 1,020 registered public functions
across 81 submodules:

```bash
python scripts/registry_stats.py --check
```

The registry and schema layer are part of the public surface. They support
programmatic discovery through `sp.list_functions()`, `sp.describe_function()`,
and `sp.function_schema()`. That schema breadth is useful for agents, but it
does not mean every registered helper is numerically validated.

## Validation Boundary

Current JSS source-snapshot audit counts: 52 `certified`, 25 `validated`, 940
`api_stable`, and 3 `experimental` registry symbols. The certified/validated
surface is therefore 77 symbols, while 751 stable auto-registered symbols
remain API-stable but not parity-backed.

The `validated` tier requires known-truth, reference-parity, external-parity,
coverage, or explicit convention evidence. Unit and regression tests support
API stability; they do not by themselves promote a function to `validated`.
`certified` is reserved for entries in the main cross-language or published
reference parity harness.

The source-snapshot evidence audit checks that all certified/validated symbols
have registry-attached evidence notes and that those notes resolve to source
files included in the JSS package. The current archive includes 133
registry-evidence source files.
The current source snapshot also tracks 135 registry-evidence source files
in the live validation-note inventory.

## Reproducible Audit Artifacts

The submission archive carries the generated audit outputs:

- `Paper-JSS/replication/results/jss_full_audit.{json,md}`
- `Paper-JSS/replication/results/claim_lint.{json,md}`
- `Paper-JSS/replication/results/validation_evidence_audit.{json,md}`
- `Paper-JSS/replication/results/source_snapshot_manifest.{json,md}`
- `Paper-JSS/replication/results/reproduction_environment_audit.md`
- `Paper-JSS/replication/results/methodological_gap_ledger.md`

The headline short path is intentionally reviewer-bounded:

```bash
cd Paper-JSS
make reproduce-tier1
make verify-submission-package
```

Tier 1 exercises Python-only smoke, registry, schema, validation-claim, formal
JSS, and package-coherence checks. The transcript explicitly states that the
Tier 1 path intentionally does not require live R or Stata; parity evidence is
shipped as source, lockfiles, scripts, and archived result artifacts.

## Parity And Replication Anchors

StatsPAI includes validation fixtures for common teaching and replication
benchmarks, including:

- Card-style returns-to-schooling IV estimates.
- LaLonde / Dehejia-Wahba job-training benchmarks.
- Lee-style close-election regression discontinuity.
- Callaway-Sant'Anna difference-in-differences examples.
- California Proposition 99 synthetic-control examples.

Known convention differences are documented in parity reports rather than
hidden. Bandwidth selectors, regularisation constants, small-sample
standard-error conventions, fold-split randomness, and identification-dependent
SCM disagreement are tracked in audit artifacts when they affect exact
numerical matching.

## Research Use

StatsPAI is being used in working-paper workflows connected to the Stanford
Rural Education Action Program and related empirical policy evaluation work. No
peer-reviewed research article using StatsPAI has yet been published. The
current impact claim is therefore based on credible near-term research use,
reproducible validation materials, public package distribution, and
reviewer-verifiable examples rather than published downstream citations.

## Public Distribution And Commercial Disclosure

StatsPAI is publicly distributed on PyPI and archived on Zenodo. Public stars,
forks, issue templates, contribution instructions, release notes, and CI checks
are community-readiness signals, not evidence of independent scholarly
adoption.

StatsPAI Inc. is the legal entity associated with the project. CoPaper.AI is a
commercial downstream product that may call the MIT-licensed StatsPAI package.
The StatsPAI package itself is permanently open source under the MIT license.
