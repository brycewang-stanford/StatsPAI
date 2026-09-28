# Pending workflow patches

Changes to `.github/workflows/*` that the Claude GitHub App cannot push
(GitHub refuses workflow edits without the `workflows` permission). Apply
locally and push from a user account:

```bash
git apply plans/pending-workflow-patches/<patch>
git add .github/workflows && git commit -m "ci: <subject>" && git push
```

Delete the patch file and its row below in the same commit.

| patch | what it does |
| --- | --- |
| _(none pending)_ | |
