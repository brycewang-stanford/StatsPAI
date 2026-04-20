"""External-parity pinned tests for canonical datasets.

Each test pins StatsPAI's numerical output on a bundled simulated
replica to 4 decimal places.  This is a drift guard against silent
regressions: any estimator refactor that changes a pinned value must
explicitly update the pin in a commit, not let it slip.

For the *underlying* claim that each replica's DGP is consistent
with the published paper on the ORIGINAL data, see
``tests/external_parity/PUBLISHED_REFERENCE_VALUES.md``.
"""
