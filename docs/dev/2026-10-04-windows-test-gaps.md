# Tests that fail on Windows

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
