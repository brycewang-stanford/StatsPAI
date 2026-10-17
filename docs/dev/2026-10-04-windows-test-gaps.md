# Tests that fail on Windows

## Update 2026-10-17

`.github/workflows/windows.yml` now runs a Windows gate on every push that
touches code (windows-latest, Python 3.10 and 3.13): every test listed
below plus the MCP stdio, path and core estimator contracts, about 700
tests in 2 to 3 minutes. Its first run (commit `6966ff63`) found the list
much shorter than on 10-03:

- All ten MCP stdio tests pass. They were fixed on main between the two
  runs; the Windows drive-letter URL bug in the data loader was one cause.
- Both path tests were real bugs, fixed: the release gate compared its
  allowlist against `src\...` paths, and `test_replay_string` looked for a
  raw path inside a `repr` literal.
- `test_example_scripts[gmethods_timevarying.py]`: the example printed
  `≈` and `—` to a cp1252 console, which crashes there for any Windows
  user. Now ASCII.
- `test_cross_validate` (R fixest skip): the skip message read a field that
  does not exist. Fixed.
- `test_nonfinite_se_warns_and_is_recorded` (3.13): a real
  platform-dependence. At the NB2 dispersion boundary `sp.zinb` inverted a
  Hessian that is singular in ln(alpha); the rounding noise came out
  negative (NaN standard error) on Linux and macOS and positive on Windows.
  The boundary is now decided from the estimate and ln_alpha's standard
  error is NaN on every platform (`tests/test_nb_dispersion_boundary.py`).
- The two parity tolerances and the GES collider test passed on 3.13; on
  3.10 the second push run failed two of the old items, both real:
  - `test_icc_matches_stata_estat_icc[bin-...]`: the observed information
    was a single central-difference stencil at h = 1e-4, whose rounding
    error depends on the platform's libm. Richardson extrapolation at
    h = 1e-3 brings the ICC standard error to 6e-10 of Stata's (it was
    7e-7 on macOS, 1.3e-6 on Windows).
  - `test_ewc_joint_test_uses_the_rescaled_f`: `sp.test` solved a
    covariance that is singular in exact arithmetic before checking that
    there were enough cosine terms; Windows' BLAS noticed the singularity
    first. The check now comes first.

A full Windows run on `9d8b3705` (dispatch, 3 h 20 min and 3 h 39 min)
passed 38,480 tests on 3.10 and 38,478 on 3.13 and failed 5 and 8. All
are handled in the commit after `968a5897`:

- `sp.ges` collider (3.13): the tie margin applied only against zero, so
  `i -> j` versus `j -> i` was still decided by rounding. Fixed; a test
  injecting 1e-13 noise into the score reproduces the old failure.
- `docs/parity.md` and `docs/evidence_inventory.json` "stale": the
  builders sorted `Path` objects, which compare case-insensitively on
  Windows. Now sorted by POSIX string (also the fixture lock and the JSS
  freeze manifest).
- EViews workbook tests: Windows' `zipfile` rewrites `\\` in member names
  both when writing the fixture and when reading, so the bug being worked
  around does not exist there. The fixture now writes real backslashes and
  the "pandas fails" assertion is POSIX-only.
- HDFE NUL truncation demo (3.13 job, pandas 3): Arrow strings keep the
  NUL; the demonstration is now limited to object strings.
- Agent-bench R oracle: the runner's R has no `EValue` / `jsonlite`; the
  test skips without them.
- ICC (3.13): already fixed in `506099db`.

These tests are now in the Windows gate.

A second full run on `4ae3af69`: Python 3.10 clean (38,490 passed); 3.13
failed 8, all handled in the next commit:

- `test_cov95_rd_misc` (`sp.rd_honest` bias bound vs RDHonest, 1.1e-7 over
  a 1e-7 budget): the bandwidth minimises a smooth criterion by comparing
  values, which places it to about sqrt(eps) on any optimiser (R's too);
  the bias bound grows like h^2. That one quantity now has 1e-6, with the
  mechanism stated; the other four stay at 1e-7.
- `test_cluster_synth_spectral_method`: the donors have no cluster
  structure, so the size of the treated unit's spectral cluster depends on
  ARPACK's eigenvectors; Windows put one donor there and the fit refused.
  The test now augments, which still runs the spectral branch.
- `test_feols_iv_first_stage_demeaner` (6 cases): not a Windows problem.
  `sp.feols` sweeps fixed effects in the written order, raw pyfixest in
  `set()` order, which follows the process's string-hash seed; with two
  effects each order stops on its own at `fixef_tol`, so the test's
  bit-for-bit comparison passed or failed with PYTHONHASHSEED on every
  platform (seeds 1, 3, 4 fail locally with the same 5.1e-13). The first
  attempt blamed the pyfixest version and was wrong (both CI legs run
  0.60.0). The test is now exact with one fixed effect and bounded by the
  sweep-order effect with two; it passes under 12 hash seeds.

The gate then passed on both versions, and a full run on `49ca1b65` was
clean on 3.10 and 3.13. Windows 3.10 and 3.13 are back in the nightly full
matrix of `ci-cd.yml` (2026-10-17). 3.11 and 3.12 have not had a full run.


Status on 2026-10-04: the Windows legs are out of the full CI matrix
(`.github/workflows/ci-cd.yml`). Nobody on the project has a Windows machine,
and a full run there takes about five hours, so these cannot be debugged by
trial and error on CI. Windows is not a tested platform until this list is
empty. Wheels for the Rust HDFE backend are still built for Windows.

The list is from the manual full run of 2026-10-03 on commit `2ed39334`
(run 37117954810): `windows-latest`, Python 3.10 to 3.13. Each leg failed 27
or 28 of about 26,500 tests. Nine of those failed on every platform and are
fixed since. The rest fail only on Windows, with one exception noted below.

## MCP server over stdio (10 tests)

These matter most. If they describe the server and not the tests, the MCP
server does not work on Windows.

- `tests/test_mcp_client_matrix.py::test_each_supported_revision_is_negotiated_and_usable`
  (three protocol revisions): `assert True is False`
- `tests/test_mcp_client_matrix.py::test_text_only_client_gets_the_whole_result_and_the_whole_error`,
  `::test_calls_work_without_an_initialize_handshake`: `KeyError: 'coefficients'`
- `tests/test_mcp_client_matrix.py::test_client_without_sampling_is_not_made_to_wait`,
  `::test_handles_do_not_survive_a_restart_and_the_error_says_so`:
  `KeyError: 'result_id'`
- `tests/test_mcp_stdio_chain.py::test_full_analysis_chain_over_stdio`,
  `tests/test_mcp_stdio_subprocess.py::test_sampling_round_trip_during_tools_call`:
  the tool call hits the 600 s timeout
- `tests/test_mcp_stdio_subprocess.py::test_stdout_is_pure_jsonrpc`:
  `assert True is False`

## Paths (2 tests)

- `tests/test_mcp_hardening.py::TestOutputShaping::test_replay_string`: the
  replay string does not contain the `C:\...` path the test looks for
- `tests/test_release_gate.py::test_no_pattern_in_the_package_changes_under_ascii_transliteration`:
  reports `src\statspai\...` paths

## Numerical (2 tests)

- `tests/reference_parity/test_iv_stata_commands_parity.py::test_factor_terms_equal_hand_made_dummies`:
  0.24682051922570736 against 0.2468205192269579, a gap of 1.25e-12 where
  the test allows 1e-12
- `tests/reference_parity/test_panel_icc_lrtest_parity.py::test_icc_matches_stata_estat_icc[bin-bin_laplace_icc]`:
  relative gap 1.7e-6 to Stata where the budget is 1e-6. This one also
  failed on Linux with Python 3.9 (an older SciPy); it passes on Linux and
  macOS with Python 3.10 and later. Neither tolerance was widened.

## Other

- `tests/test_cross_validate.py::TestEnginesHonourRequestedVariance::test_r_fixest_clusters_on_the_requested_variable`:
  `AttributeError: 'EngineEstimate' object has no attribute 'error'`
- `tests/test_example_scripts.py::test_example_script_runs[gmethods_timevarying.py]`
- `tests/reference_parity/test_ges_parity.py::test_collider_no_spurious_edge_between_parents`
  (Python 3.12 and 3.13 only): the collider comes back unoriented
- `tests/test_fail_loudly_contracts.py::test_nonfinite_se_warns_and_is_recorded`
  (3.10 and 3.11 only): no `ConvergenceWarning`
- `tests/test_regress_hac_options.py::test_ewc_joint_test_uses_the_rescaled_f`
  (3.10 only): the expected message does not match

## Putting Windows back

Add the four `windows-latest` entries to the matrix in `ci-cd.yml` and run
the workflow by hand with `test_scope=full`.
