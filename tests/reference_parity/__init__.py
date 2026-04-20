"""Reference parity test suite.

Tests in this directory validate StatsPAI estimators against:
1. **Recovery tests**: deterministic DGPs with known population parameters;
   estimates must recover truth within 3 standard errors.
2. **Cross-estimator parity**: estimators that should agree on a given DGP
   must agree numerically (e.g., classic 2x2 DID == CS2021 on a 2-period
   2-group DGP).
3. **Pinned outputs**: fixed-seed DGPs with hardcoded numerical outputs
   as drift guards.

Every numerical reference is sourced in ``REFERENCES.md``.
"""
