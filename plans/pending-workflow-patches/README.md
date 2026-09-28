# Pending workflow patches

Changes to `.github/workflows/*` that the Claude GitHub App cannot push
(GitHub refuses workflow edits without the `workflows` permission). Apply
locally and push from a user account:

```bash
git apply plans/pending-workflow-patches/<patch>
git add .github/workflows && git commit -m "ci: <subject>" && git push
```

Delete the patch file in the same commit.

| patch | what it does |
| --- | --- |
| `2026-09-28-ci-cd-mcp-gate.patch` | adds the MCP protocol / tool-contract / subprocess suites and `tests/test_agent_native_gaps.py` to the push-time fast gate; wheel smoke test launches `statspai-mcp --help` and checks the bundled schema snapshot |
